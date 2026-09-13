//! Composable sample pipelines.
//!
//! A [`DataPipeline`] is a list of [`Stage`]s. The first stage ignores its
//! input and produces samples — a shard list, say — and each later stage
//! transforms the stream. This mirrors the Python `DataPipeline` class, and it
//! exists for the same reason: a pipeline has to be re-runnable, because
//! `repeat` and `with_epoch` need to start the source over again.
//!
//! ```
//! use webdataset::pipeline::DataPipeline;
//! use webdataset::shardlists::SimpleShardList;
//! use webdataset::sources::ShardsToSamples;
//!
//! let pipeline = DataPipeline::new()
//!     .with(SimpleShardList::new(["../../testdata/sample.tgz"])?)
//!     .with(ShardsToSamples::streaming());
//!
//! let count = pipeline.iter().filter(Result::is_ok).count();
//! assert!(count > 0);
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::sync::Arc;

use webdataset_core::error::Result;
use webdataset_core::sample::Sample;

/// A stream of samples, each of which may have failed.
pub type SampleStream = Box<dyn Iterator<Item = Result<Sample>> + Send>;

/// One step of a pipeline.
///
/// `apply` is called once per epoch, so a stage that carries state across
/// epochs — an epoch counter for deterministic shuffling, for instance — needs
/// interior mutability.
pub trait Stage: Send + Sync + std::fmt::Debug {
    /// Transform the incoming stream. Source stages ignore `input`.
    fn apply(&self, input: SampleStream) -> SampleStream;
}

impl<T: Stage + ?Sized> Stage for Arc<T> {
    fn apply(&self, input: SampleStream) -> SampleStream {
        (**self).apply(input)
    }
}

/// An empty stream, the input handed to a pipeline's first stage.
pub fn empty_stream() -> SampleStream {
    Box::new(std::iter::empty())
}

/// A stage built from a closure.
pub struct FnStage<F> {
    name: &'static str,
    apply: F,
}

impl<F> std::fmt::Debug for FnStage<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl<F> Stage for FnStage<F>
where
    F: Fn(SampleStream) -> SampleStream + Send + Sync,
{
    fn apply(&self, input: SampleStream) -> SampleStream {
        (self.apply)(input)
    }
}

/// Wrap a closure as a [`Stage`].
pub fn stage_fn<F>(name: &'static str, apply: F) -> FnStage<F>
where
    F: Fn(SampleStream) -> SampleStream + Send + Sync,
{
    FnStage { name, apply }
}

/// A source stage that replays a fixed list of samples.
#[derive(Debug, Clone)]
pub struct Samples(Vec<Sample>);

impl Samples {
    /// Yield these samples, once per epoch.
    pub fn new(samples: impl IntoIterator<Item = Sample>) -> Samples {
        Samples(samples.into_iter().collect())
    }
}

impl Stage for Samples {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        Box::new(self.0.clone().into_iter().map(Ok))
    }
}

/// A list of stages that can be run, and re-run, as a stream of samples.
///
/// Cloning a pipeline is cheap and shares stage state, so a cloned pipeline can
/// be handed to another thread — that is how [`DataLoader`](crate::DataLoader)
/// fans one pipeline out over several workers.
#[derive(Debug, Clone, Default)]
pub struct DataPipeline {
    stages: Vec<Arc<dyn Stage>>,
    repetitions: Repetitions,
    limit: Option<usize>,
}

/// How many times a pipeline replays its source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Repetitions {
    /// Run the source once, as an ordinary dataset does.
    #[default]
    Once,
    /// Run the source a fixed number of times.
    Times(usize),
    /// Run the source until the consumer stops asking.
    Forever,
}

impl DataPipeline {
    /// An empty pipeline.
    pub fn new() -> DataPipeline {
        DataPipeline::default()
    }

    /// Append a stage, consuming and returning the pipeline.
    pub fn with(mut self, stage: impl Stage + 'static) -> DataPipeline {
        self.push(stage);
        self
    }

    /// Append a stage in place.
    pub fn push(&mut self, stage: impl Stage + 'static) {
        self.stages.push(Arc::new(stage));
    }

    /// Append an already shared stage in place.
    pub fn push_shared(&mut self, stage: Arc<dyn Stage>) {
        self.stages.push(stage);
    }

    /// The number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Whether the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Borrow stage `index`.
    pub fn stage(&self, index: usize) -> Option<&Arc<dyn Stage>> {
        self.stages.get(index)
    }

    /// Replay the source `epochs` times.
    pub fn repeat(mut self, epochs: usize) -> DataPipeline {
        self.repetitions = Repetitions::Times(epochs);
        self
    }

    /// Replay the source indefinitely.
    pub fn repeat_forever(mut self) -> DataPipeline {
        self.repetitions = Repetitions::Forever;
        self
    }

    /// Make an epoch exactly `nsamples` long, replaying the source as needed.
    ///
    /// This does not change how many distinct samples exist — it only redraws
    /// the epoch boundary, which is what fixed-length training loops want.
    ///
    /// Note that a [`DataLoader`](crate::DataLoader) runs one copy of the
    /// pipeline per worker, so the limit applies per worker.
    pub fn with_epoch(mut self, nsamples: usize) -> DataPipeline {
        self.repetitions = Repetitions::Forever;
        self.limit = Some(nsamples);
        self
    }

    /// Stop after `nsamples` samples in total.
    pub fn take(mut self, nsamples: usize) -> DataPipeline {
        self.limit = Some(nsamples);
        self
    }

    /// Run the stages once, from the source onwards.
    fn run_once(&self) -> SampleStream {
        let mut stream = empty_stream();
        for stage in &self.stages {
            stream = stage.apply(stream);
        }
        stream
    }

    /// Build a fresh stream of samples.
    ///
    /// Calling this twice runs the source twice; stages such as the shard
    /// shuffler advance their epoch each time.
    pub fn iter(&self) -> SampleStream {
        let stream: SampleStream = match self.repetitions {
            Repetitions::Once => self.run_once(),
            _ => Box::new(Repeated::new(self.clone())),
        };
        match self.limit {
            Some(n) => Box::new(stream.take(n)),
            None => stream,
        }
    }

    /// Iterate, discarding failures rather than reporting them.
    ///
    /// Convenient in examples and tests; production code should look at the
    /// errors, or install a [`Handler`](webdataset_core::Handler) that does.
    pub fn iter_ok(&self) -> impl Iterator<Item = Sample> + Send {
        self.iter().filter_map(|r| match r {
            Ok(sample) => Some(sample),
            Err(e) => {
                log::warn!("dropping sample: {e}");
                None
            }
        })
    }
}

impl IntoIterator for &DataPipeline {
    type Item = Result<Sample>;
    type IntoIter = SampleStream;

    fn into_iter(self) -> SampleStream {
        self.iter()
    }
}

/// Replays a pipeline's source, epoch after epoch.
struct Repeated {
    pipeline: DataPipeline,
    current: Option<SampleStream>,
    epochs_left: Option<usize>,
    produced_this_epoch: usize,
}

impl Repeated {
    fn new(pipeline: DataPipeline) -> Repeated {
        let epochs_left = match pipeline.repetitions {
            Repetitions::Once => Some(1),
            Repetitions::Times(n) => Some(n),
            Repetitions::Forever => None,
        };
        Repeated { pipeline, current: None, epochs_left, produced_this_epoch: 0 }
    }
}

impl Iterator for Repeated {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        loop {
            if self.current.is_none() {
                match self.epochs_left {
                    Some(0) => return None,
                    Some(n) => self.epochs_left = Some(n - 1),
                    None => {}
                }
                self.current = Some(self.pipeline.run_once());
                self.produced_this_epoch = 0;
            }
            let stream = self.current.as_mut().expect("just installed");
            match stream.next() {
                Some(item) => {
                    self.produced_this_epoch += 1;
                    return Some(item);
                }
                None => {
                    self.current = None;
                    // An empty epoch would otherwise spin forever.
                    if self.produced_this_epoch == 0 {
                        return None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use webdataset_core::value::Value;

    fn three() -> Samples {
        Samples::new((0..3).map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s
        }))
    }

    fn keys(pipeline: &DataPipeline, limit: usize) -> Vec<String> {
        pipeline.iter().take(limit).map(|s| s.unwrap().key().unwrap().to_string()).collect()
    }

    #[test]
    fn runs_the_stages_in_order() {
        let pipeline = DataPipeline::new().with(three()).with(stage_fn("double", |input| {
            Box::new(input.map(|item| {
                item.map(|mut s| {
                    let doubled = s.get("cls").and_then(Value::as_i64).unwrap_or(0) * 2;
                    s.insert("cls", Value::Int(doubled));
                    s
                })
            }))
        }));
        let values: Vec<i64> = pipeline.iter().map(|s| s.unwrap().get("cls").unwrap().as_i64().unwrap()).collect();
        assert_eq!(values, [0, 2, 4]);
    }

    #[test]
    fn replays_the_source_on_every_iteration() {
        let pipeline = DataPipeline::new().with(three());
        assert_eq!(keys(&pipeline, 10).len(), 3);
        assert_eq!(keys(&pipeline, 10).len(), 3, "a second pass must see the same samples");
    }

    #[test]
    fn repeats_a_fixed_number_of_epochs() {
        let pipeline = DataPipeline::new().with(three()).repeat(3);
        assert_eq!(pipeline.iter().count(), 9);
    }

    #[test]
    fn redraws_epoch_boundaries() {
        let pipeline = DataPipeline::new().with(three()).with_epoch(7);
        let seen = keys(&pipeline, 100);
        assert_eq!(seen.len(), 7);
        assert_eq!(seen[0], "k0");
        assert_eq!(seen[3], "k0", "the source restarts once exhausted");
    }

    #[test]
    fn stops_instead_of_spinning_on_an_empty_source() {
        let pipeline = DataPipeline::new().with(Samples::new([])).with_epoch(10);
        assert_eq!(pipeline.iter().count(), 0);
    }

    #[test]
    fn truncates_with_take() {
        let pipeline = DataPipeline::new().with(three()).take(2);
        assert_eq!(pipeline.iter().count(), 2);
    }

    #[test]
    fn is_shared_when_cloned() {
        let pipeline = DataPipeline::new().with(three());
        let clone = pipeline.clone();
        let handle = std::thread::spawn(move || clone.iter().count());
        assert_eq!(handle.join().unwrap(), 3);
        assert_eq!(pipeline.iter().count(), 3);
    }
}
