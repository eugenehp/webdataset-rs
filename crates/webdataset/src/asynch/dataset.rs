//! The asynchronous counterpart of [`WebDataset`](crate::WebDataset).

use std::sync::Arc;

use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{HandlerRef, reraise_exception};
use webdataset_core::sample::Sample;
use webdataset_core::value::Value;
use webdataset_shard::Selection;

use crate::dataset::NodeSplit;
use crate::decode::Decoder;
use crate::shardlists::{ResampledShards, SimpleShardList};

use super::filters::AsyncSampleStreamExt;
use super::pipeline::{AsyncDataPipeline, AsyncStage, LiftSource, SampleStream, stage_fn};
use super::sources::{AsyncOpener, AsyncShardsToSamples, FileOpener};
use super::stages::{CheckEmpty, Shuffle, SingleNodeOnly, SplitByNode, SplitByWorker};

/// Where an asynchronous dataset's shards come from.
#[derive(Debug, Clone)]
enum SourceSpec {
    /// Brace patterns and `::` lists to expand.
    Patterns(Vec<String>),
    /// URLs to use exactly as given.
    Verbatim(Vec<String>),
}

/// Builds an [`AsyncWebDataset`].
#[derive(Debug)]
pub struct AsyncWebDatasetBuilder {
    source: SourceSpec,
    opener: Option<Arc<dyn AsyncOpener>>,
    concurrency: usize,
    shard_shuffle: Option<usize>,
    resampled: bool,
    node_split: NodeSplit,
    worker_split: bool,
    selection: Selection,
    empty_check: bool,
    seed: Option<u64>,
    deterministic: bool,
    handler: HandlerRef,
}

impl AsyncWebDatasetBuilder {
    fn new(source: SourceSpec) -> AsyncWebDatasetBuilder {
        AsyncWebDatasetBuilder {
            source,
            opener: None,
            concurrency: 1,
            shard_shuffle: None,
            resampled: false,
            node_split: NodeSplit::Refuse,
            worker_split: true,
            selection: Selection::new(),
            empty_check: true,
            seed: None,
            deterministic: false,
            handler: reraise_exception(),
        }
    }

    /// Fetch shards with `opener`; the local filesystem is the default.
    pub fn opener(mut self, opener: Arc<dyn AsyncOpener>) -> AsyncWebDatasetBuilder {
        self.opener = Some(opener);
        self
    }

    /// Keep this many shards in flight at once.
    ///
    /// Above one, samples from different shards interleave in an unspecified
    /// order — which is the point, since that is what hides fetch latency.
    pub fn concurrency(mut self, concurrency: usize) -> AsyncWebDatasetBuilder {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Shuffle the shard list through a buffer of this size each epoch.
    pub fn shard_shuffle(mut self, bufsize: usize) -> AsyncWebDatasetBuilder {
        self.shard_shuffle = Some(bufsize);
        self
    }

    /// Draw shards with replacement instead of partitioning them.
    pub fn resampled(mut self, resampled: bool) -> AsyncWebDatasetBuilder {
        self.resampled = resampled;
        self
    }

    /// Choose how shards are divided between distributed processes.
    pub fn node_split(mut self, split: NodeSplit) -> AsyncWebDatasetBuilder {
        self.node_split = split;
        self
    }

    /// Whether to divide shards between loader workers. On by default.
    pub fn worker_split(mut self, split: bool) -> AsyncWebDatasetBuilder {
        self.worker_split = split;
        self
    }

    /// Choose or rename the files read out of each shard.
    pub fn selection(mut self, selection: Selection) -> AsyncWebDatasetBuilder {
        self.selection = selection;
        self
    }

    /// Whether to fail when the pipeline yields nothing. On by default.
    pub fn empty_check(mut self, check: bool) -> AsyncWebDatasetBuilder {
        self.empty_check = check;
        self
    }

    /// Seed the shard shuffle and resampling.
    pub fn seed(mut self, seed: u64) -> AsyncWebDatasetBuilder {
        self.seed = Some(seed);
        self
    }

    /// Make shuffling reproducible across runs while still varying per epoch.
    pub fn deterministic(mut self, deterministic: bool) -> AsyncWebDatasetBuilder {
        self.deterministic = deterministic;
        self
    }

    /// Decide what happens when a shard or sample cannot be read.
    pub fn handler(mut self, handler: HandlerRef) -> AsyncWebDatasetBuilder {
        self.handler = handler;
        self
    }

    /// Assemble the pipeline.
    pub fn build(self) -> Result<AsyncWebDataset> {
        let seed = self.seed.unwrap_or_else(|| {
            std::env::var("WDS_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
                use rand::RngExt;
                rand::rng().random()
            })
        });

        let mut pipeline = AsyncDataPipeline::new();

        // 1. the shard list, which is pure computation and so is reused
        match &self.source {
            SourceSpec::Patterns(patterns) if self.resampled => pipeline.push(LiftSource::new(
                ResampledShards::new(patterns)?.with_seed(seed).deterministic(self.deterministic),
            )),
            SourceSpec::Verbatim(urls) if self.resampled => pipeline
                .push(LiftSource::new(ResampledShards::new(urls)?.with_seed(seed).deterministic(self.deterministic))),
            SourceSpec::Patterns(patterns) => pipeline.push(LiftSource::new(SimpleShardList::new(patterns)?)),
            SourceSpec::Verbatim(urls) => pipeline.push(LiftSource::new(SimpleShardList::verbatim(urls.clone()))),
        }

        // 2. distribute shards across nodes and workers
        match self.node_split {
            NodeSplit::Refuse if !self.resampled => pipeline.push(SingleNodeOnly),
            NodeSplit::ByNode => pipeline.push(SplitByNode),
            _ => {}
        }
        if self.worker_split && !self.resampled {
            pipeline.push(SplitByWorker);
        }

        // 3. shuffle the shard order
        if let Some(bufsize) = self.shard_shuffle {
            let shuffle = Shuffle::new(bufsize);
            pipeline.push(match self.deterministic {
                true => shuffle.deterministic(seed),
                false => shuffle.with_seed(seed),
            });
        }

        // 4. fetch the shards and group their files into samples
        let opener = self.opener.unwrap_or_else(|| Arc::new(FileOpener::new()));
        pipeline.push(
            AsyncShardsToSamples::new(opener)
                .with_selection(self.selection)
                .with_handler(self.handler.clone())
                .with_concurrency(self.concurrency),
        );

        if self.empty_check {
            pipeline.push(CheckEmpty::default());
        }

        Ok(AsyncWebDataset { pipeline, handler: self.handler, seed })
    }
}

/// A dataset read from WebDataset-format shards, without blocking.
#[derive(Debug, Clone)]
pub struct AsyncWebDataset {
    pipeline: AsyncDataPipeline,
    handler: HandlerRef,
    seed: u64,
}

impl AsyncWebDataset {
    /// Start building a dataset from brace patterns or a `::`-separated list.
    pub fn builder(urls: impl AsRef<str>) -> AsyncWebDatasetBuilder {
        AsyncWebDatasetBuilder::new(SourceSpec::Patterns(vec![urls.as_ref().to_string()]))
    }

    /// Start building a dataset from several patterns.
    pub fn builder_from<S: AsRef<str>>(patterns: impl IntoIterator<Item = S>) -> AsyncWebDatasetBuilder {
        AsyncWebDatasetBuilder::new(SourceSpec::Patterns(
            patterns.into_iter().map(|p| p.as_ref().to_string()).collect(),
        ))
    }

    /// Start building a dataset from URLs that need no expansion.
    pub fn builder_verbatim<S: Into<String>>(urls: impl IntoIterator<Item = S>) -> AsyncWebDatasetBuilder {
        AsyncWebDatasetBuilder::new(SourceSpec::Verbatim(urls.into_iter().map(Into::into).collect()))
    }

    /// Wrap an already assembled pipeline.
    pub fn from_pipeline(pipeline: AsyncDataPipeline) -> AsyncWebDataset {
        AsyncWebDataset { pipeline, handler: reraise_exception(), seed: 0 }
    }

    /// The seed the shard shuffle and resampling were built with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The underlying pipeline.
    pub fn pipeline(&self) -> &AsyncDataPipeline {
        &self.pipeline
    }

    /// Take the underlying pipeline.
    pub fn into_pipeline(self) -> AsyncDataPipeline {
        self.pipeline
    }

    /// Append an arbitrary stage.
    pub fn with(mut self, stage: impl AsyncStage + 'static) -> AsyncWebDataset {
        self.pipeline.push(stage);
        self
    }

    /// Shuffle samples through a buffer of `bufsize`.
    ///
    /// Seeded exactly as [`WebDataset::shuffle`](crate::WebDataset::shuffle)
    /// is, so a blocking and an asynchronous dataset built with the same seed
    /// shuffle the same way.
    pub fn shuffle(self, bufsize: usize) -> AsyncWebDataset {
        let seed = self.seed;
        self.with(Shuffle::new(bufsize).deterministic(seed))
    }

    /// Decode every field of every sample.
    pub fn decode(self, decoder: Decoder) -> AsyncWebDataset {
        let decoder = Arc::new(decoder);
        let handler = self.handler.clone();
        self.with(stage_fn("decode", move |input: SampleStream| {
            let decoder = decoder.clone();
            Box::pin(input.map_sample_with(move |sample| decoder.decode(sample).map(Some), handler.clone()))
        }))
    }

    /// Decode with the default handler chain.
    pub fn decode_basic(self) -> AsyncWebDataset {
        self.decode(Decoder::default())
    }

    /// Decode images with the given `imagespec`, plus the default handlers.
    #[cfg(feature = "image")]
    pub fn decode_images(self, imagespec: &str) -> Result<AsyncWebDataset> {
        let handler = crate::images::ImageHandler::parse(imagespec)?;
        Ok(self.decode(Decoder::new(vec![Arc::new(handler)])))
    }

    /// Apply a function to each sample; returning `None` drops it.
    pub fn map(self, f: impl Fn(Sample) -> Result<Option<Sample>> + Send + Sync + 'static) -> AsyncWebDataset {
        let f = Arc::new(f);
        let handler = self.handler.clone();
        self.with(stage_fn("map", move |input: SampleStream| {
            let f = f.clone();
            Box::pin(input.map_sample_with(move |sample| f(sample), handler.clone()))
        }))
    }

    /// Apply a function to one field of each sample.
    pub fn map_field(
        self,
        field: impl Into<String>,
        f: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> AsyncWebDataset {
        let field = field.into();
        self.map(move |mut sample| {
            let Some(value) = sample.remove(&field) else {
                return Err(Error::MissingKey {
                    wanted: vec![field.clone()],
                    available: sample.keys().map(str::to_string).collect(),
                });
            };
            sample.insert(field.clone(), f(value)?);
            Ok(Some(sample))
        })
    }

    /// Keep the samples a predicate accepts.
    pub fn select(self, predicate: impl Fn(&Sample) -> bool + Send + Sync + 'static) -> AsyncWebDataset {
        let predicate = Arc::new(predicate);
        self.with(stage_fn("select", move |input: SampleStream| {
            let predicate = predicate.clone();
            Box::pin(input.select(move |sample| predicate(sample)))
        }))
    }

    /// Make an epoch exactly `nsamples` long, replaying the source as needed.
    pub fn with_epoch(mut self, nsamples: usize) -> AsyncWebDataset {
        self.pipeline = self.pipeline.with_epoch(nsamples);
        self
    }

    /// Replay the dataset `epochs` times.
    pub fn repeat(mut self, epochs: usize) -> AsyncWebDataset {
        self.pipeline = self.pipeline.repeat(epochs);
        self
    }

    /// Stop after `nsamples` in total.
    pub fn take(mut self, nsamples: usize) -> AsyncWebDataset {
        self.pipeline = self.pipeline.take(nsamples);
        self
    }

    /// Start an epoch.
    pub fn stream(&self) -> SampleStream {
        self.pipeline.stream()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asynch::MemoryOpener;
    use futures_executor::block_on;
    use futures_util::{StreamExt, TryStreamExt};

    fn shard(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
        std::fs::read(path).expect("reading the shard")
    }

    fn testdata(name: &str) -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    fn dataset() -> AsyncWebDataset {
        AsyncWebDataset::builder_verbatim([testdata("imagenet-000000.tgz")]).build().expect("building")
    }

    fn collect(dataset: &AsyncWebDataset) -> Vec<Sample> {
        block_on(dataset.stream().try_collect::<Vec<_>>()).expect("no failures")
    }

    #[test]
    fn reads_a_shard_end_to_end() {
        let samples = collect(&dataset());
        assert_eq!(samples.len(), 47);
        assert!(samples[0].contains_key("png"));
        assert!(samples[0].contains_key("cls"));
    }

    #[test]
    fn matches_the_blocking_dataset() {
        let blocking: Vec<Sample> = crate::WebDataset::builder_verbatim([testdata("imagenet-000000.tgz")])
            .build()
            .expect("building")
            .decode_basic()
            .iter()
            .map(|s| s.expect("a sample"))
            .collect();

        let asynchronous = collect(&dataset().decode_basic());
        assert_eq!(asynchronous, blocking, "the two pipelines must agree sample for sample");
    }

    #[test]
    fn decodes_and_projects() {
        use crate::asynch::AsyncSampleStreamExt;

        let rows: Vec<Vec<Value>> =
            block_on(dataset().decode_basic().stream().to_tuple(["png", "cls"]).try_collect()).expect("no failures");

        assert_eq!(rows.len(), 47);
        assert!(rows[0][0].as_bytes().is_some());
        assert_eq!(rows[0][1].as_i64(), Some(304));
    }

    #[test]
    fn shuffles_reproducibly_for_a_fixed_seed() {
        let keys = |seed: u64| -> Vec<String> {
            collect(
                &AsyncWebDataset::builder_verbatim([testdata("imagenet-000000.tgz")])
                    .seed(seed)
                    .build()
                    .expect("building")
                    .shuffle(20),
            )
            .into_iter()
            .map(|s| s.key().expect("a key").to_string())
            .collect()
        };
        assert_eq!(keys(1), keys(1));
        assert_ne!(keys(1), keys(2));
    }

    #[test]
    fn maps_and_selects() {
        let samples = collect(
            &dataset()
                .decode_basic()
                .select(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(-1) >= 0)
                .map_field("cls", |v| Ok(Value::Int(v.as_i64().unwrap_or(0) + 1000))),
        );
        assert_eq!(samples.len(), 47);
        assert!(samples.iter().all(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(0) >= 1000));
    }

    #[test]
    fn honours_epoch_length() {
        assert_eq!(collect(&dataset().with_epoch(100)).len(), 100);
    }

    #[test]
    fn fetches_several_shards_at_once() {
        let opener = MemoryOpener::new()
            .with("mem://a.tar", shard("sample.tgz"))
            .with("mem://b.tar", shard("imagenet-000000.tgz"));

        let dataset = AsyncWebDataset::builder_verbatim(["mem://a.tar", "mem://b.tar"])
            .opener(Arc::new(opener))
            .concurrency(2)
            .build()
            .expect("building");

        assert_eq!(collect(&dataset).len(), 90 + 47);
    }

    #[test]
    fn reports_an_empty_dataset() {
        let starved = AsyncWebDataset::builder_verbatim([testdata("sample.tgz")])
            .selection(Selection::new().select(|_| false))
            .build()
            .expect("building");
        let outcome = block_on(starved.stream().collect::<Vec<_>>());
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]), "{outcome:?}");
    }

    #[test]
    fn resamples_shards_endlessly() {
        let dataset = AsyncWebDataset::builder_verbatim([testdata("sample.tgz")])
            .resampled(true)
            .build()
            .expect("building")
            .with_epoch(500);
        assert_eq!(collect(&dataset).len(), 500);
    }
}
