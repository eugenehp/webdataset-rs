//! Running a pipeline over several worker threads.
//!
//! Decoding images and running augmentations is compute-bound, so a single
//! thread reading one shard at a time will not keep a GPU busy. [`DataLoader`]
//! runs one copy of the pipeline per worker, each seeing a different slice of
//! the shard list, and merges their output.
//!
//! Splitting is the pipeline's job, not the loader's: put a
//! [`SplitByWorker`](crate::shardlists::SplitByWorker) stage after the shard
//! list and each worker will claim a disjoint set of shards. The loader sets
//! the worker identity that stage reads.
//!
//! ```
//! use webdataset::loader::DataLoader;
//! use webdataset::pipeline::{DataPipeline, Samples};
//! use webdataset::shardlists::SplitByWorker;
//! use webdataset_core::Sample;
//!
//! let pipeline = DataPipeline::new()
//!     .with(Samples::new((0..100).map(|i| Sample::with_key(format!("k{i}")))))
//!     .with(SplitByWorker);
//!
//! let loader = DataLoader::new(pipeline).with_workers(4);
//! assert_eq!(loader.iter().count(), 100, "every sample is seen exactly once");
//! ```
//!
//! Samples from different workers interleave in an unspecified order, so a
//! loader with more than one worker is not reproducible sample-for-sample even
//! with fixed seeds.
//!
//! Without the `threads` feature — on WebAssembly, for instance — the loader
//! still works but always runs inline on the calling thread, whatever worker
//! count you ask for.

use webdataset_core::error::Result;
use webdataset_core::sample::Sample;

use crate::pipeline::DataPipeline;

#[cfg(feature = "threads")]
use {
    std::sync::Arc,
    std::sync::atomic::{AtomicBool, Ordering},
    std::sync::mpsc::{Receiver, SyncSender, sync_channel},
    std::thread::JoinHandle,
    webdataset_core::utils::with_worker,
};

/// Runs a pipeline over a pool of worker threads.
#[derive(Debug, Clone)]
pub struct DataLoader {
    pipeline: DataPipeline,
    workers: usize,
    prefetch: usize,
}

impl DataLoader {
    /// Load from `pipeline` on the calling thread.
    pub fn new(pipeline: DataPipeline) -> DataLoader {
        DataLoader { pipeline, workers: 0, prefetch: 8 }
    }

    /// Use `workers` threads; zero or one runs inline, without a channel.
    ///
    /// Without the `threads` feature this is recorded but not acted on: the
    /// pipeline always runs inline.
    pub fn with_workers(mut self, workers: usize) -> DataLoader {
        self.workers = workers;
        self
    }

    /// Allow this many samples per worker to be buffered ahead of the consumer.
    pub fn with_prefetch(mut self, prefetch: usize) -> DataLoader {
        self.prefetch = prefetch.max(1);
        self
    }

    /// How many worker threads will be started.
    pub fn workers(&self) -> usize {
        self.workers
    }

    /// The pipeline being run.
    pub fn pipeline(&self) -> &DataPipeline {
        &self.pipeline
    }

    /// Start an epoch.
    pub fn iter(&self) -> LoaderIter {
        #[cfg(not(feature = "threads"))]
        return LoaderIter { inner: Inner::Inline(self.pipeline.iter()) };

        #[cfg(feature = "threads")]
        self.spawn()
    }

    /// Start an epoch over the worker pool.
    #[cfg(feature = "threads")]
    fn spawn(&self) -> LoaderIter {
        if self.workers <= 1 {
            return LoaderIter { inner: Inner::Inline(self.pipeline.iter()) };
        }

        let stop = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = sync_channel(self.prefetch * self.workers);
        let mut handles = Vec::with_capacity(self.workers);

        for index in 0..self.workers {
            let pipeline = self.pipeline.clone();
            let sender: SyncSender<Result<Sample>> = sender.clone();
            let stop = stop.clone();
            let workers = self.workers;
            let handle = std::thread::Builder::new()
                .name(format!("webdataset-worker-{index}"))
                .spawn(move || {
                    with_worker(index, workers, || {
                        for item in pipeline.iter() {
                            if stop.load(Ordering::Relaxed) || sender.send(item).is_err() {
                                break;
                            }
                        }
                    })
                })
                .expect("failed to start a loader worker");
            handles.push(handle);
        }
        // Drop the template sender so the channel closes once the workers do.
        drop(sender);

        LoaderIter { inner: Inner::Threaded { receiver, handles, stop } }
    }

    /// Iterate, discarding failures rather than reporting them.
    pub fn iter_ok(&self) -> impl Iterator<Item = Sample> {
        self.iter().filter_map(|item| match item {
            Ok(sample) => Some(sample),
            Err(e) => {
                log::warn!("dropping sample: {e}");
                None
            }
        })
    }
}

/// One epoch of a [`DataLoader`].
pub struct LoaderIter {
    inner: Inner,
}

enum Inner {
    Inline(crate::pipeline::SampleStream),
    #[cfg(feature = "threads")]
    Threaded {
        receiver: Receiver<Result<Sample>>,
        handles: Vec<JoinHandle<()>>,
        stop: Arc<AtomicBool>,
    },
}

impl Iterator for LoaderIter {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        match &mut self.inner {
            Inner::Inline(stream) => stream.next(),
            #[cfg(feature = "threads")]
            Inner::Threaded { receiver, .. } => receiver.recv().ok(),
        }
    }
}

#[cfg(feature = "threads")]
impl Drop for LoaderIter {
    fn drop(&mut self) {
        let Inner::Threaded { receiver, handles, stop } = &mut self.inner else {
            return;
        };
        // Ask the workers to stop, then drain so any that are blocked on a full
        // channel can make the one more send they need to notice.
        stop.store(true, Ordering::Relaxed);
        while receiver.recv().is_ok() {}
        for handle in handles.drain(..) {
            if handle.join().is_err() {
                log::error!("a loader worker panicked");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Samples;
    use crate::shardlists::SplitByWorker;
    use crate::stages::MapStage;
    use webdataset_core::error::Error;

    fn pipeline(n: usize) -> DataPipeline {
        DataPipeline::new().with(Samples::new((0..n).map(|i| Sample::with_key(format!("k{i}")))))
    }

    fn split_pipeline(n: usize) -> DataPipeline {
        pipeline(n).with(SplitByWorker)
    }

    #[test]
    fn runs_inline_with_one_worker() {
        let loader = DataLoader::new(pipeline(10));
        assert_eq!(loader.iter().count(), 10);
        assert_eq!(loader.with_workers(1).iter().count(), 10);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn splits_work_across_threads_without_duplication() {
        let loader = DataLoader::new(split_pipeline(200)).with_workers(4);
        let mut keys: Vec<String> = loader.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();

        assert_eq!(keys.len(), 200);
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 200, "no sample should be seen twice");
    }

    #[cfg(feature = "threads")]
    #[test]
    fn runs_the_pipeline_once_per_worker_without_a_splitter() {
        let loader = DataLoader::new(pipeline(10)).with_workers(3);
        assert_eq!(loader.iter().count(), 30, "without a splitter every worker reads everything");
    }

    #[test]
    fn can_be_iterated_repeatedly() {
        let loader = DataLoader::new(split_pipeline(50)).with_workers(2);
        assert_eq!(loader.iter().count(), 50);
        assert_eq!(loader.iter().count(), 50);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn stops_the_workers_when_dropped_early() {
        let pipeline = DataPipeline::new()
            .with(Samples::new((0..10_000).map(|i| Sample::with_key(format!("k{i}")))))
            .with(SplitByWorker);

        let loader = DataLoader::new(pipeline).with_workers(4).with_prefetch(2);
        let taken: Vec<_> = loader.iter().take(5).collect();
        assert_eq!(taken.len(), 5);
        // Dropping the iterator joins the workers; reaching here means it did
        // not deadlock on a full channel.
    }

    #[test]
    fn reports_errors_from_workers() {
        let pipeline = split_pipeline(20)
            .with(MapStage::new(|_| -> Result<Option<Sample>> { Err(Error::value("worker failure")) }));
        let loader = DataLoader::new(pipeline).with_workers(2);
        let outcome: Vec<_> = loader.iter().collect();
        assert_eq!(outcome.len(), 20);
        assert!(outcome.iter().all(Result::is_err));
    }

    #[test]
    fn iter_ok_drops_failures() {
        let pipeline = split_pipeline(20).with(MapStage::new(|s: Sample| {
            let n: usize = s.key().unwrap().trim_start_matches('k').parse().unwrap();
            if n % 2 == 0 { Err(Error::value("even")) } else { Ok(Some(s)) }
        }));
        let loader = DataLoader::new(pipeline).with_workers(2);
        assert_eq!(loader.iter_ok().count(), 10);
    }
}
