//! Stages for the asynchronous pipeline.
//!
//! These are the few stages that sit *between* the shard list and the shard
//! reader, so they have to work on a stream rather than an iterator. Each one
//! is a handful of lines, and each behaves exactly as its blocking counterpart
//! in [`stages`](crate::stages) does.
//!
//! Everything else is either a source — lifted with
//! [`LiftSource`](super::LiftSource), because building a shard list is pure
//! computation — or an adapter on [`super::AsyncSampleStreamExt`].

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures_util::stream::{self, StreamExt};
use webdataset_core::error::Error;
use webdataset_core::utils::{make_seed, worker_info};

use super::filters::AsyncSampleStreamExt;
use super::pipeline::{AsyncStage, SampleStream};

/// Keeps only the shards belonging to this distributed rank.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitByNode;

impl AsyncStage for SplitByNode {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let info = worker_info();
        if info.world_size <= 1 {
            return input;
        }
        Box::pin(deal(input, info.rank, info.world_size))
    }
}

/// Keeps only the shards belonging to this loader worker.
#[derive(Debug, Clone, Copy, Default)]
pub struct SplitByWorker;

impl AsyncStage for SplitByWorker {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let info = worker_info();
        if info.num_workers <= 1 {
            return input;
        }
        Box::pin(deal(input, info.worker, info.num_workers))
    }
}

/// Keep every `total`th item, starting at `index`.
fn deal(
    input: SampleStream,
    index: usize,
    total: usize,
) -> impl futures_core::Stream<Item = webdataset_core::error::Result<webdataset_core::Sample>> + Send {
    input
        .enumerate()
        .filter(move |(position, _)| {
            let keep = position % total == index;
            async move { keep }
        })
        .map(|(_, item)| item)
}

/// Fails loudly when a pipeline that has no node splitter is run distributed.
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleNodeOnly;

impl AsyncStage for SingleNodeOnly {
    fn apply(&self, input: SampleStream) -> SampleStream {
        if worker_info().world_size > 1 {
            return Box::pin(stream::once(async {
                Err(Error::value(
                    "this pipeline has no node splitter but is running on multiple nodes; \
                     add SplitByNode, or resample shards instead",
                ))
            }));
        }
        input
    }
}

/// Shuffles samples through an in-memory buffer.
///
/// Seeding works as it does for the blocking
/// [`Shuffle`](crate::stages::Shuffle): no seed means fresh entropy each epoch,
/// a fixed seed means the same order every epoch, and `deterministic` mixes in
/// the epoch and the worker so runs reproduce without epochs repeating.
#[derive(Debug)]
pub struct Shuffle {
    bufsize: usize,
    seed: Option<u64>,
    deterministic: bool,
    epoch: AtomicUsize,
}

impl Shuffle {
    /// Shuffle through a buffer of `bufsize` samples.
    pub fn new(bufsize: usize) -> Shuffle {
        Shuffle { bufsize, seed: None, deterministic: false, epoch: AtomicUsize::new(0) }
    }

    /// Use a fixed seed, giving the same order in every epoch.
    pub fn with_seed(mut self, seed: u64) -> Shuffle {
        self.seed = Some(seed);
        self
    }

    /// Derive each epoch's seed from `seed`, the epoch, and the worker.
    pub fn deterministic(mut self, seed: u64) -> Shuffle {
        self.seed = Some(seed);
        self.deterministic = true;
        self
    }
}

impl AsyncStage for Shuffle {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) as u64;
        let seed = self.seed.map(|seed| match self.deterministic {
            true => make_seed(&[seed, epoch, worker_info().seed()]),
            false => seed,
        });
        input.shuffled(self.bufsize, seed)
    }
}

/// Fails if the pipeline produced no samples at all.
#[derive(Debug, Clone)]
pub struct CheckEmpty {
    message: Arc<str>,
}

impl Default for CheckEmpty {
    fn default() -> CheckEmpty {
        CheckEmpty {
            message: Arc::from(
                "no samples found; you may have fewer shards than workers. \
                 Disable this check with empty_check(false).",
            ),
        }
    }
}

impl CheckEmpty {
    /// Fail with this message instead of the default one.
    pub fn with_message(message: impl Into<Arc<str>>) -> CheckEmpty {
        CheckEmpty { message: message.into() }
    }
}

impl AsyncStage for CheckEmpty {
    fn apply(&self, input: SampleStream) -> SampleStream {
        input.non_empty(self.message.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asynch::AsyncDataPipeline;
    use crate::pipeline::Samples;
    use futures_executor::block_on;
    use futures_util::TryStreamExt;
    use webdataset_core::Sample;
    use webdataset_core::utils::with_worker;

    fn eight() -> Samples {
        Samples::new((0..8).map(|i| Sample::with_key(format!("k{i}"))))
    }

    fn keys(pipeline: &AsyncDataPipeline) -> Vec<String> {
        block_on(pipeline.stream().try_collect::<Vec<_>>())
            .expect("no failures")
            .into_iter()
            .map(|s| s.key().expect("a key").to_string())
            .collect()
    }

    #[test]
    fn splits_shards_across_workers() {
        let mut seen = Vec::new();
        for worker in 0..4 {
            let assigned =
                with_worker(worker, 4, || keys(&AsyncDataPipeline::new().with_source(eight()).with(SplitByWorker)));
            assert_eq!(assigned.len(), 2, "worker {worker}");
            seen.extend(assigned);
        }
        seen.sort();
        assert_eq!(seen.len(), 8, "every shard goes to exactly one worker");
        seen.dedup();
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn splitting_is_a_no_op_with_one_worker() {
        assert_eq!(keys(&AsyncDataPipeline::new().with_source(eight()).with(SplitByWorker)).len(), 8);
    }

    #[test]
    fn splits_the_same_way_the_blocking_stage_does() {
        for worker in 0..4 {
            let asynchronous =
                with_worker(worker, 4, || keys(&AsyncDataPipeline::new().with_source(eight()).with(SplitByWorker)));

            let blocking: Vec<String> = with_worker(worker, 4, || {
                crate::pipeline::DataPipeline::new()
                    .with(eight())
                    .with(crate::shardlists::SplitByWorker)
                    .iter()
                    .map(|s| s.expect("a sample").key().expect("a key").to_string())
                    .collect()
            });

            assert_eq!(asynchronous, blocking, "worker {worker}");
        }
    }

    #[test]
    fn shuffle_varies_between_epochs_when_deterministic() {
        let pipeline = AsyncDataPipeline::new().with_source(eight()).with(Shuffle::new(4).deterministic(7));
        assert_ne!(keys(&pipeline), keys(&pipeline));
    }

    #[test]
    fn shuffle_repeats_with_a_fixed_seed() {
        let build = || AsyncDataPipeline::new().with_source(eight()).with(Shuffle::new(4).with_seed(3));
        assert_eq!(keys(&build()), keys(&build()));
    }

    #[test]
    fn check_empty_reports_a_starved_pipeline() {
        let empty = AsyncDataPipeline::new().with_source(Samples::new([])).with(CheckEmpty::default());
        let outcome = block_on(empty.stream().collect::<Vec<_>>());
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]));

        let full = AsyncDataPipeline::new().with_source(eight()).with(CheckEmpty::default());
        assert_eq!(keys(&full).len(), 8);
    }

    #[test]
    fn single_node_only_passes_a_single_node_through() {
        assert_eq!(keys(&AsyncDataPipeline::new().with_source(eight()).with(SingleNodeOnly)).len(), 8);
    }
}
