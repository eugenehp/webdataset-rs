//! The high-level entry point: [`WebDataset`].
//!
//! `WebDataset` assembles the standard pipeline — shard list, node and worker
//! split, shard shuffle, archive reading, sample grouping — and then lets you
//! append the rest with a fluid interface. It is the Rust counterpart of the
//! Python class of the same name.
//!
//! ```no_run
//! use webdataset::{Decoder, WebDataset};
//! use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
//!
//! let dataset = WebDataset::builder("https://host/imagenet-{000000..000146}.tar")
//!     .shard_shuffle(100)
//!     .cache_dir("./_cache")
//!     .build()?
//!     .shuffle(1000)
//!     .decode(Decoder::default());
//!
//! for batch in dataset.iter().to_tuple(["jpg;png", "cls"]).batched(32, true) {
//!     let batch = batch?;
//!     // batch[0] holds the images, batch[1] the labels
//!     # let _ = batch;
//! }
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! ## Choosing how shards are distributed
//!
//! With `resampled(true)` each worker draws shards with replacement, so no
//! worker ever runs dry and epoch length is whatever you ask for with
//! [`with_epoch`](WebDataset::with_epoch). Otherwise shards are dealt out
//! round-robin, which is exact but needs at least as many shards as workers.
//! Without either, a pipeline run on several nodes refuses to start rather than
//! silently training every node on the same data.

use std::path::PathBuf;
use std::sync::Arc;

use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{HandlerRef, reraise_exception};
use webdataset_core::sample::Sample;
use webdataset_core::value::Value;
use webdataset_io::cache::FileCache;
use webdataset_shard::reader::Selection;

use crate::decode::Decoder;
use crate::loader::DataLoader;
use crate::pipeline::{DataPipeline, SampleStream, Stage};
use crate::shardlists::{ResampledShards, SimpleShardList, SingleNodeOnly, SplitByNode, SplitByWorker};
use crate::sources::{CachingOpener, Opener, ShardsToSamples, StreamingOpener};
use crate::stages::{CheckEmpty, Decode, MapStage, Rename, SelectStage, Shuffle, Slice};

/// How shards are divided between distributed processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeSplit {
    /// Refuse to run on more than one node without an explicit choice.
    #[default]
    Refuse,
    /// Deal shards out round-robin across ranks.
    ByNode,
    /// Do nothing; every node sees every shard.
    None,
}

/// Where a dataset's shards come from.
#[derive(Debug, Clone)]
enum SourceSpec {
    /// Brace patterns and `::` lists to expand.
    Patterns(Vec<String>),
    /// URLs to use exactly as given.
    Verbatim(Vec<String>),
    /// A YAML multi-source specification.
    #[cfg(feature = "yaml")]
    Yaml(String),
}

/// Builds a [`WebDataset`].
#[derive(Debug)]
pub struct WebDatasetBuilder {
    source: SourceSpec,
    shard_shuffle: Option<usize>,
    resampled: bool,
    cache_dir: Option<PathBuf>,
    cache_size: Option<u64>,
    node_split: NodeSplit,
    worker_split: bool,
    selection: Selection,
    empty_check: bool,
    seed: Option<u64>,
    deterministic: bool,
    handler: HandlerRef,
}

impl WebDatasetBuilder {
    fn new(source: SourceSpec) -> WebDatasetBuilder {
        WebDatasetBuilder {
            source,
            shard_shuffle: None,
            resampled: false,
            cache_dir: None,
            cache_size: None,
            node_split: NodeSplit::Refuse,
            worker_split: true,
            selection: Selection::new(),
            empty_check: true,
            seed: None,
            deterministic: false,
            handler: reraise_exception(),
        }
    }

    /// Shuffle the shard list through a buffer of this size each epoch.
    ///
    /// Shard-level shuffling is what decorrelates the data; the sample-level
    /// [`shuffle`](WebDataset::shuffle) then mixes within that window.
    pub fn shard_shuffle(mut self, bufsize: usize) -> WebDatasetBuilder {
        self.shard_shuffle = Some(bufsize);
        self
    }

    /// Draw shards with replacement instead of partitioning them.
    pub fn resampled(mut self, resampled: bool) -> WebDatasetBuilder {
        self.resampled = resampled;
        self
    }

    /// Cache shards in this directory.
    ///
    /// The directory must already exist, so that a typo does not silently
    /// scatter gigabytes into an unexpected place.
    pub fn cache_dir(mut self, directory: impl Into<PathBuf>) -> WebDatasetBuilder {
        self.cache_dir = Some(directory.into());
        self
    }

    /// Evict cached shards once the cache exceeds this many bytes.
    pub fn cache_size(mut self, bytes: u64) -> WebDatasetBuilder {
        self.cache_size = Some(bytes);
        self
    }

    /// Choose how shards are divided between distributed processes.
    pub fn node_split(mut self, split: NodeSplit) -> WebDatasetBuilder {
        self.node_split = split;
        self
    }

    /// Whether to divide shards between loader workers. On by default.
    pub fn worker_split(mut self, split: bool) -> WebDatasetBuilder {
        self.worker_split = split;
        self
    }

    /// Choose or rename the files read out of each shard.
    pub fn selection(mut self, selection: Selection) -> WebDatasetBuilder {
        self.selection = selection;
        self
    }

    /// Whether to fail when the pipeline yields nothing. On by default.
    pub fn empty_check(mut self, check: bool) -> WebDatasetBuilder {
        self.empty_check = check;
        self
    }

    /// Seed the shard shuffle and resampling.
    pub fn seed(mut self, seed: u64) -> WebDatasetBuilder {
        self.seed = Some(seed);
        self
    }

    /// Make shuffling reproducible across runs while still varying per epoch.
    pub fn deterministic(mut self, deterministic: bool) -> WebDatasetBuilder {
        self.deterministic = deterministic;
        self
    }

    /// Decide what happens when a shard or sample cannot be read.
    pub fn handler(mut self, handler: HandlerRef) -> WebDatasetBuilder {
        self.handler = handler;
        self
    }

    /// Assemble the pipeline.
    pub fn build(self) -> Result<WebDataset> {
        let seed = self
            .seed
            .unwrap_or_else(|| std::env::var("WDS_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or_else(rand_seed));

        let mut pipeline = DataPipeline::new();

        // 1. the shard list
        match &self.source {
            #[cfg(feature = "yaml")]
            SourceSpec::Yaml(text) => {
                let sample = crate::shardlists::MultiShardSample::from_yaml(text)?;
                sample.set_seed(seed);
                pipeline.push(sample);
            }
            SourceSpec::Patterns(patterns) if self.resampled => {
                pipeline.push(ResampledShards::new(patterns)?.with_seed(seed).deterministic(self.deterministic));
            }
            SourceSpec::Verbatim(urls) if self.resampled => {
                pipeline.push(ResampledShards::new(urls)?.with_seed(seed).deterministic(self.deterministic));
            }
            SourceSpec::Patterns(patterns) => pipeline.push(SimpleShardList::new(patterns)?),
            SourceSpec::Verbatim(urls) => pipeline.push(SimpleShardList::verbatim(urls.clone())),
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

        // 4. read the shards and group their files into samples
        let opener: Arc<dyn Opener> = match &self.cache_dir {
            Some(directory) => {
                if !directory.exists() {
                    return Err(Error::value(format!("cache directory {} does not exist", directory.display())));
                }
                let mut cache = FileCache::new(directory.clone());
                if let Some(size) = self.cache_size {
                    cache = cache.with_budget(size, Some(std::time::Duration::from_secs(30)));
                }
                Arc::new(CachingOpener::with_cache(Arc::new(cache)))
            }
            None => Arc::new(StreamingOpener),
        };
        pipeline.push(ShardsToSamples::new(opener).with_selection(self.selection).with_handler(self.handler.clone()));

        if self.empty_check {
            pipeline.push(CheckEmpty::default());
        }

        Ok(WebDataset { pipeline, handler: self.handler, seed })
    }
}

fn rand_seed() -> u64 {
    use rand::RngExt;
    rand::rng().random()
}

/// A dataset read from WebDataset-format shards.
///
/// Build one with [`WebDataset::builder`], then append stages with the methods
/// below. Anything that changes the item type — batching, projecting to tuples
/// — happens on [`iter`](WebDataset::iter) using the adapters in
/// [`filters`](crate::filters).
#[derive(Debug, Clone)]
pub struct WebDataset {
    pipeline: DataPipeline,
    handler: HandlerRef,
    seed: u64,
}

impl WebDataset {
    /// Start building a dataset from brace patterns or a `::`-separated list.
    pub fn builder(urls: impl AsRef<str>) -> WebDatasetBuilder {
        WebDatasetBuilder::new(SourceSpec::Patterns(vec![urls.as_ref().to_string()]))
    }

    /// Start building a dataset from several patterns.
    pub fn builder_from<S: AsRef<str>>(patterns: impl IntoIterator<Item = S>) -> WebDatasetBuilder {
        WebDatasetBuilder::new(SourceSpec::Patterns(patterns.into_iter().map(|p| p.as_ref().to_string()).collect()))
    }

    /// Start building a dataset from URLs that need no expansion.
    pub fn builder_verbatim<S: Into<String>>(urls: impl IntoIterator<Item = S>) -> WebDatasetBuilder {
        WebDatasetBuilder::new(SourceSpec::Verbatim(urls.into_iter().map(Into::into).collect()))
    }

    /// Start building a dataset from a YAML multi-source specification.
    #[cfg(feature = "yaml")]
    pub fn builder_from_yaml(spec: impl Into<String>) -> WebDatasetBuilder {
        WebDatasetBuilder::new(SourceSpec::Yaml(spec.into()))
    }

    /// Build a dataset from `urls` with the default settings.
    ///
    /// Shards are not shuffled and the dataset refuses to run distributed; use
    /// [`builder`](WebDataset::builder) to change either.
    pub fn open(urls: impl AsRef<str>) -> Result<WebDataset> {
        WebDataset::builder(urls).build()
    }

    /// Wrap an already assembled pipeline.
    pub fn from_pipeline(pipeline: DataPipeline) -> WebDataset {
        WebDataset { pipeline, handler: reraise_exception(), seed: 0 }
    }

    /// The seed the shard shuffle and resampling were built with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The underlying pipeline.
    pub fn pipeline(&self) -> &DataPipeline {
        &self.pipeline
    }

    /// Take the underlying pipeline.
    pub fn into_pipeline(self) -> DataPipeline {
        self.pipeline
    }

    /// Append an arbitrary stage.
    pub fn with(mut self, stage: impl Stage + 'static) -> WebDataset {
        self.pipeline.push(stage);
        self
    }

    /// Shuffle samples through a buffer of `bufsize`.
    pub fn shuffle(self, bufsize: usize) -> WebDataset {
        let seed = self.seed;
        self.with(Shuffle::new(bufsize).deterministic(seed))
    }

    /// Decode every field of every sample.
    pub fn decode(self, decoder: Decoder) -> WebDataset {
        let handler = self.handler.clone();
        self.with(Decode::new(decoder).with_handler(handler))
    }

    /// Decode with the default handler chain.
    pub fn decode_basic(self) -> WebDataset {
        self.decode(Decoder::default())
    }

    /// Decode images with the given `imagespec`, plus the default handlers.
    #[cfg(feature = "image")]
    pub fn decode_images(self, imagespec: &str) -> Result<WebDataset> {
        let handler = crate::images::ImageHandler::parse(imagespec)?;
        Ok(self.decode(Decoder::new(vec![Arc::new(handler)])))
    }

    /// Apply a function to each sample; returning `None` drops it.
    pub fn map(self, f: impl Fn(Sample) -> Result<Option<Sample>> + Send + Sync + 'static) -> WebDataset {
        let handler = self.handler.clone();
        self.with(MapStage::new(f).with_handler(handler))
    }

    /// Apply a function to one field of each sample.
    pub fn map_field(
        self,
        field: impl Into<String>,
        f: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> WebDataset {
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
    pub fn select(self, predicate: impl Fn(&Sample) -> bool + Send + Sync + 'static) -> WebDataset {
        self.with(SelectStage::new(predicate))
    }

    /// Rename fields, resolving each source from a `;`-separated alternation.
    pub fn rename<A: Into<String>, B: Into<String>>(self, renames: impl IntoIterator<Item = (A, B)>) -> WebDataset {
        let handler = self.handler.clone();
        self.with(Rename::new(renames).with_handler(handler))
    }

    /// Take `count` samples starting at `start`.
    pub fn slice(self, start: usize, count: Option<usize>) -> WebDataset {
        self.with(Slice::new(start, count))
    }

    /// Make an epoch exactly `nsamples` long, replaying the source as needed.
    ///
    /// Each [`DataLoader`] worker runs its own copy of the pipeline, so the
    /// limit applies per worker: with four workers, `with_epoch(1000)` yields
    /// 4000 samples per epoch. Divide by the worker count if you want the
    /// total to match.
    pub fn with_epoch(mut self, nsamples: usize) -> WebDataset {
        self.pipeline = self.pipeline.with_epoch(nsamples);
        self
    }

    /// Replay the dataset `epochs` times.
    pub fn repeat(mut self, epochs: usize) -> WebDataset {
        self.pipeline = self.pipeline.repeat(epochs);
        self
    }

    /// Stop after `nsamples` in total.
    pub fn take(mut self, nsamples: usize) -> WebDataset {
        self.pipeline = self.pipeline.take(nsamples);
        self
    }

    /// Start an epoch.
    pub fn iter(&self) -> SampleStream {
        self.pipeline.iter()
    }

    /// Start an epoch, discarding failures.
    pub fn iter_ok(&self) -> impl Iterator<Item = Sample> + Send {
        self.pipeline.iter_ok()
    }

    /// Read this dataset with a pool of worker threads.
    pub fn loader(&self) -> DataLoader {
        DataLoader::new(self.pipeline.clone())
    }
}

impl IntoIterator for &WebDataset {
    type Item = Result<Sample>;
    type IntoIter = SampleStream;

    fn into_iter(self) -> SampleStream {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::SampleIteratorExt;

    fn testdata(name: &str) -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    fn dataset() -> WebDataset {
        WebDataset::builder_verbatim([testdata("imagenet-000000.tgz")]).build().unwrap()
    }

    #[test]
    fn reads_a_shard_end_to_end() {
        let samples: Vec<Sample> = dataset().iter().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), 47);
        assert!(samples[0].contains_key("png"));
        assert!(samples[0].contains_key("cls"));
    }

    #[test]
    fn decodes_and_projects() {
        let rows: Vec<Vec<Value>> =
            dataset().decode_basic().iter().to_tuple(["png", "cls"]).map(|r| r.unwrap()).take(4).collect();

        assert_eq!(rows.len(), 4);
        assert!(rows[0][0].as_bytes().is_some(), "png stays raw without an image handler");
        assert!(rows[0][1].as_i64().is_some(), "cls decodes to an integer");
    }

    #[test]
    fn shuffles_reproducibly_for_a_fixed_seed() {
        let keys = |seed: u64| -> Vec<String> {
            WebDataset::builder_verbatim([testdata("imagenet-000000.tgz")])
                .seed(seed)
                .deterministic(true)
                .build()
                .unwrap()
                .shuffle(20)
                .iter()
                .map(|s| s.unwrap().key().unwrap().to_string())
                .collect()
        };
        assert_eq!(keys(1), keys(1));
        assert_ne!(keys(1), keys(2));
    }

    #[test]
    fn maps_and_selects() {
        let samples: Vec<Sample> = dataset()
            .decode_basic()
            .select(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(-1) >= 0)
            .map_field("cls", |v| Ok(Value::Int(v.as_i64().unwrap_or(0) + 1000)))
            .iter()
            .map(|s| s.unwrap())
            .collect();

        assert_eq!(samples.len(), 47);
        assert!(samples.iter().all(|s| s.get("cls").unwrap().as_i64().unwrap() >= 1000));
    }

    #[test]
    fn renames_fields() {
        let sample = dataset().rename([("image", "png;jpg"), ("label", "cls")]).iter().next().unwrap().unwrap();
        assert!(sample.contains_key("image"));
        assert!(sample.contains_key("label"));
        assert!(!sample.contains_key("png"));
    }

    #[test]
    fn honours_epoch_length() {
        let dataset = dataset().with_epoch(100);
        assert_eq!(dataset.iter().count(), 100, "the source replays to fill the epoch");
    }

    #[test]
    fn batches_through_the_iterator_adapters() {
        let batches: Vec<Sample> = dataset().decode_basic().iter().batched(16, true).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3, "47 samples in batches of 16");
        assert_eq!(batches[0].get("cls").unwrap().as_tensor().unwrap().shape(), &[16]);
    }

    #[test]
    fn refuses_a_missing_cache_directory() {
        let err = WebDataset::builder_verbatim([testdata("sample.tgz")])
            .cache_dir("/definitely/not/here")
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"), "{err}");
    }

    #[test]
    fn reports_a_dataset_that_yields_no_samples() {
        // The check sits directly after shard reading, so it catches a starved
        // source — the usual cause is having fewer shards than workers.
        let starved =
            || WebDataset::builder_verbatim([testdata("sample.tgz")]).selection(Selection::new().select(|_| false));

        let outcome: Vec<_> = starved().build().unwrap().iter().collect();
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]), "{outcome:?}");

        assert_eq!(starved().empty_check(false).build().unwrap().iter().count(), 0);
    }

    #[test]
    fn filtering_everything_out_downstream_is_not_an_error() {
        // The check runs before user stages, so a selective pipeline that keeps
        // nothing is simply empty rather than a failure.
        let dataset = WebDataset::builder_verbatim([testdata("sample.tgz")]).build().unwrap().select(|_| false);
        assert_eq!(dataset.iter().count(), 0);
    }

    #[test]
    fn resamples_shards_endlessly() {
        let dataset =
            WebDataset::builder_verbatim([testdata("sample.tgz")]).resampled(true).build().unwrap().with_epoch(500);
        assert_eq!(dataset.iter().count(), 500);
    }

    #[test]
    fn runs_over_a_worker_pool() {
        let dataset =
            WebDataset::builder_verbatim([testdata("sample.tgz"), testdata("imagenet-000000.tgz")]).build().unwrap();
        let single = dataset.iter().count();
        let pooled = dataset.loader().with_workers(2).iter().count();
        assert_eq!(pooled, single, "splitting by worker must not change the sample count");
    }

    #[cfg(feature = "image")]
    #[test]
    fn decodes_images() {
        let sample = dataset().decode_images("rgb8").unwrap().iter().next().unwrap().unwrap();
        let image = sample.get("png").unwrap().as_tensor().expect("png should decode to a tensor");
        assert_eq!(image.shape().len(), 3);
        assert_eq!(image.shape()[2], 3, "rgb8 produces three channels");
    }
}
