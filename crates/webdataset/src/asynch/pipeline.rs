//! The asynchronous counterpart of [`DataPipeline`](crate::pipeline::DataPipeline).

use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use futures_util::stream::{self, StreamExt};
use webdataset_core::error::Result;
use webdataset_core::sample::Sample;

use crate::pipeline::{Repetitions, Stage};

/// A stream of samples, each of which may have failed.
pub type SampleStream = Pin<Box<dyn Stream<Item = Result<Sample>> + Send>>;

/// One step of an asynchronous pipeline.
///
/// As in the blocking pipeline, `apply` runs once per epoch, so a stage that
/// carries state across epochs needs interior mutability.
pub trait AsyncStage: Send + Sync + std::fmt::Debug {
    /// Transform the incoming stream. Source stages ignore `input`.
    fn apply(&self, input: SampleStream) -> SampleStream;
}

/// An empty stream, the input handed to a pipeline's first stage.
pub fn empty_stream() -> SampleStream {
    Box::pin(stream::empty())
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

impl<F> AsyncStage for FnStage<F>
where
    F: Fn(SampleStream) -> SampleStream + Send + Sync,
{
    fn apply(&self, input: SampleStream) -> SampleStream {
        (self.apply)(input)
    }
}

/// Wrap a closure as an [`AsyncStage`].
pub fn stage_fn<F>(name: &'static str, apply: F) -> FnStage<F>
where
    F: Fn(SampleStream) -> SampleStream + Send + Sync,
{
    FnStage { name, apply }
}

/// Runs a blocking source stage and presents its output as a stream.
///
/// Shard lists are pure computation — expanding brace patterns, dealing shards
/// out round-robin, drawing them with replacement — so the blocking
/// implementations are reused rather than written twice. Anything that reads
/// from the network belongs in [`AsyncShardsToSamples`](super::AsyncShardsToSamples)
/// instead.
#[derive(Debug, Clone)]
pub struct LiftSource(Arc<dyn Stage>);

impl LiftSource {
    /// Lift a blocking source stage, such as a shard list.
    pub fn new(stage: impl Stage + 'static) -> LiftSource {
        LiftSource(Arc::new(stage))
    }

    /// Lift an already shared source stage.
    pub fn shared(stage: Arc<dyn Stage>) -> LiftSource {
        LiftSource(stage)
    }
}

impl AsyncStage for LiftSource {
    fn apply(&self, _input: SampleStream) -> SampleStream {
        Box::pin(stream::iter(self.0.apply(crate::pipeline::empty_stream())))
    }
}

/// A list of stages that can be run, and re-run, as a stream of samples.
#[derive(Debug, Clone, Default)]
pub struct AsyncDataPipeline {
    stages: Vec<Arc<dyn AsyncStage>>,
    repetitions: Repetitions,
    limit: Option<usize>,
}

impl AsyncDataPipeline {
    /// An empty pipeline.
    pub fn new() -> AsyncDataPipeline {
        AsyncDataPipeline::default()
    }

    /// Append a stage, consuming and returning the pipeline.
    pub fn with(mut self, stage: impl AsyncStage + 'static) -> AsyncDataPipeline {
        self.push(stage);
        self
    }

    /// Append a blocking source stage, such as a shard list.
    pub fn with_source(self, stage: impl Stage + 'static) -> AsyncDataPipeline {
        self.with(LiftSource::new(stage))
    }

    /// Append a stage in place.
    pub fn push(&mut self, stage: impl AsyncStage + 'static) {
        self.stages.push(Arc::new(stage));
    }

    /// Append an already shared stage in place.
    pub fn push_shared(&mut self, stage: Arc<dyn AsyncStage>) {
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

    /// Replay the source `epochs` times.
    pub fn repeat(mut self, epochs: usize) -> AsyncDataPipeline {
        self.repetitions = Repetitions::Times(epochs);
        self
    }

    /// Replay the source indefinitely.
    pub fn repeat_forever(mut self) -> AsyncDataPipeline {
        self.repetitions = Repetitions::Forever;
        self
    }

    /// Make an epoch exactly `nsamples` long, replaying the source as needed.
    pub fn with_epoch(mut self, nsamples: usize) -> AsyncDataPipeline {
        self.repetitions = Repetitions::Forever;
        self.limit = Some(nsamples);
        self
    }

    /// Stop after `nsamples` samples in total.
    pub fn take(mut self, nsamples: usize) -> AsyncDataPipeline {
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
    pub fn stream(&self) -> SampleStream {
        let stream = match self.repetitions {
            Repetitions::Once => self.run_once(),
            _ => self.repeated(),
        };
        match self.limit {
            Some(n) => Box::pin(stream.take(n)),
            None => stream,
        }
    }

    /// Replay the source, epoch after epoch, stopping if one comes back empty.
    fn repeated(&self) -> SampleStream {
        let epochs_left = match self.repetitions {
            Repetitions::Once => Some(1),
            Repetitions::Times(n) => Some(n),
            Repetitions::Forever => None,
        };

        /// What the unfold carries between epochs.
        struct State {
            pipeline: AsyncDataPipeline,
            current: SampleStream,
            epochs_left: Option<usize>,
            produced: usize,
        }

        let state = State {
            current: self.run_once(),
            pipeline: self.clone(),
            epochs_left: epochs_left.map(|n| n.saturating_sub(1)),
            produced: 0,
        };

        Box::pin(stream::unfold(Some(state), |carried| async move {
            let mut state = carried?;
            loop {
                if let Some(item) = state.current.next().await {
                    state.produced += 1;
                    return Some((item, Some(state)));
                }
                // An empty epoch would otherwise spin forever.
                if state.produced == 0 {
                    return None;
                }
                match state.epochs_left {
                    Some(0) => return None,
                    Some(n) => state.epochs_left = Some(n - 1),
                    None => {}
                }
                state.current = state.pipeline.run_once();
                state.produced = 0;
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Samples;
    use futures_executor::block_on;
    use futures_util::TryStreamExt;
    use webdataset_core::value::Value;

    fn three() -> Samples {
        Samples::new((0..3).map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s
        }))
    }

    fn keys(pipeline: &AsyncDataPipeline) -> Vec<String> {
        block_on(async {
            pipeline
                .stream()
                .try_collect::<Vec<_>>()
                .await
                .expect("no failures")
                .into_iter()
                .map(|s| s.key().expect("a key").to_string())
                .collect()
        })
    }

    #[test]
    fn lifts_a_blocking_source() {
        let pipeline = AsyncDataPipeline::new().with_source(three());
        assert_eq!(keys(&pipeline), ["k0", "k1", "k2"]);
    }

    #[test]
    fn runs_the_stages_in_order() {
        let pipeline = AsyncDataPipeline::new().with_source(three()).with(stage_fn("double", |input| {
            Box::pin(input.map(|item| {
                item.map(|mut s| {
                    let doubled = s.get("cls").and_then(Value::as_i64).unwrap_or(0) * 2;
                    s.insert("cls", Value::Int(doubled));
                    s
                })
            }))
        }));
        let values: Vec<i64> = block_on(async {
            pipeline
                .stream()
                .try_collect::<Vec<_>>()
                .await
                .expect("no failures")
                .iter()
                .map(|s| s.get("cls").and_then(Value::as_i64).expect("cls"))
                .collect()
        });
        assert_eq!(values, [0, 2, 4]);
    }

    #[test]
    fn replays_the_source_on_every_stream() {
        let pipeline = AsyncDataPipeline::new().with_source(three());
        assert_eq!(keys(&pipeline).len(), 3);
        assert_eq!(keys(&pipeline).len(), 3);
    }

    #[test]
    fn repeats_a_fixed_number_of_epochs() {
        let pipeline = AsyncDataPipeline::new().with_source(three()).repeat(3);
        assert_eq!(keys(&pipeline).len(), 9);
    }

    #[test]
    fn redraws_epoch_boundaries() {
        let pipeline = AsyncDataPipeline::new().with_source(three()).with_epoch(7);
        let seen = keys(&pipeline);
        assert_eq!(seen.len(), 7);
        assert_eq!(seen[3], "k0", "the source restarts once exhausted");
    }

    #[test]
    fn stops_instead_of_spinning_on_an_empty_source() {
        let pipeline = AsyncDataPipeline::new().with_source(Samples::new([])).with_epoch(10);
        assert_eq!(keys(&pipeline).len(), 0);
    }

    #[test]
    fn truncates_with_take() {
        let pipeline = AsyncDataPipeline::new().with_source(three()).take(2);
        assert_eq!(keys(&pipeline).len(), 2);
    }
}
