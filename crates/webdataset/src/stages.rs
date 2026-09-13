//! [`Stage`] wrappers around the filters.
//!
//! The adapters in [`filters`](crate::filters) are plain iterator adapters; the
//! stages here are the same transformations packaged so they can be appended to
//! a [`DataPipeline`](crate::pipeline::DataPipeline) and re-run every epoch.
//!
//! ```
//! use webdataset::pipeline::{DataPipeline, Samples};
//! use webdataset::stages::{MapStage, Shuffle};
//! use webdataset_core::{Sample, Value};
//!
//! let samples = (0..100).map(|i| {
//!     let mut s = Sample::with_key(format!("k{i}"));
//!     s.insert("cls", Value::Int(i));
//!     s
//! });
//!
//! let pipeline = DataPipeline::new()
//!     .with(Samples::new(samples))
//!     .with(Shuffle::new(20).with_seed(42))
//!     .with(MapStage::new(|mut s: Sample| {
//!         s.insert("seen", Value::Bool(true));
//!         Ok(Some(s))
//!     }));
//!
//! assert_eq!(pipeline.iter().count(), 100);
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{HandlerRef, reraise_exception};
use webdataset_core::sample::Sample;
use webdataset_core::utils::{make_seed, worker_info};
use webdataset_core::value::Value;

use crate::decode::Decoder;
use crate::filters::{SampleIteratorExt, rename_fields, rename_keys_in};
use crate::pipeline::{SampleStream, Stage};

/// Shuffles samples through an in-memory buffer.
///
/// Without a seed, each epoch is shuffled from fresh entropy. With
/// [`with_seed`](Shuffle::with_seed) the order is fixed and repeats every
/// epoch; [`deterministic`](Shuffle::deterministic) keeps reproducibility while
/// still varying between epochs and workers, which is what training runs want.
#[derive(Debug)]
pub struct Shuffle {
    bufsize: usize,
    initial: Option<usize>,
    seed: Option<u64>,
    deterministic: bool,
    epoch: AtomicUsize,
}

impl Shuffle {
    /// Shuffle through a buffer of `bufsize` samples.
    pub fn new(bufsize: usize) -> Shuffle {
        Shuffle { bufsize, initial: None, seed: None, deterministic: false, epoch: AtomicUsize::new(0) }
    }

    /// Start emitting once this many samples are buffered.
    pub fn with_initial(mut self, initial: usize) -> Shuffle {
        self.initial = Some(initial);
        self
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

impl Stage for Shuffle {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed) as u64;
        let seed = self.seed.map(|seed| match self.deterministic {
            true => make_seed(&[seed, epoch, worker_info().seed()]),
            false => seed,
        });
        let initial = self.initial.unwrap_or_else(|| self.bufsize.div_ceil(10).max(1));
        Box::new(crate::filters::Shuffle::new(input, self.bufsize, initial, seed))
    }
}

/// Decodes every field of every sample.
#[derive(Debug)]
pub struct Decode {
    decoder: Arc<Decoder>,
    handler: HandlerRef,
}

impl Decode {
    /// Decode with `decoder`.
    pub fn new(decoder: Decoder) -> Decode {
        Decode { decoder: Arc::new(decoder), handler: reraise_exception() }
    }

    /// Decode with the default handler chain.
    pub fn basic() -> Decode {
        Decode::new(Decoder::default())
    }

    /// Decide what happens when a sample fails to decode.
    pub fn with_handler(mut self, handler: HandlerRef) -> Decode {
        self.handler = handler;
        self
    }
}

impl Stage for Decode {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let decoder = self.decoder.clone();
        Box::new(input.map_sample_with(move |sample| decoder.decode(sample).map(Some), self.handler.clone()))
    }
}

/// Applies a function to each sample.
pub struct MapStage {
    f: Arc<dyn Fn(Sample) -> Result<Option<Sample>> + Send + Sync>,
    handler: HandlerRef,
}

impl std::fmt::Debug for MapStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MapStage")
    }
}

impl MapStage {
    /// Map each sample; returning `None` drops it.
    pub fn new(f: impl Fn(Sample) -> Result<Option<Sample>> + Send + Sync + 'static) -> MapStage {
        MapStage { f: Arc::new(f), handler: reraise_exception() }
    }

    /// Decide what happens when the function fails.
    pub fn with_handler(mut self, handler: HandlerRef) -> MapStage {
        self.handler = handler;
        self
    }
}

impl Stage for MapStage {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let f = self.f.clone();
        Box::new(input.map_sample_with(move |sample| f(sample), self.handler.clone()))
    }
}

/// Keeps the samples a predicate accepts.
pub struct SelectStage {
    predicate: Arc<dyn Fn(&Sample) -> bool + Send + Sync>,
}

impl std::fmt::Debug for SelectStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SelectStage")
    }
}

impl SelectStage {
    /// Keep the samples for which `predicate` returns true.
    pub fn new(predicate: impl Fn(&Sample) -> bool + Send + Sync + 'static) -> SelectStage {
        SelectStage { predicate: Arc::new(predicate) }
    }
}

impl Stage for SelectStage {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let predicate = self.predicate.clone();
        Box::new(input.select(move |sample| predicate(sample)))
    }
}

/// Renames fields, resolving each source from a `;`-separated alternation.
#[derive(Debug)]
pub struct Rename {
    renames: Vec<(String, String)>,
    keep: bool,
    handler: HandlerRef,
}

impl Rename {
    /// Rename `from` to `to` for each pair, keeping the other fields.
    ///
    /// ```
    /// use webdataset::stages::Rename;
    ///
    /// // `image` is taken from whichever of png, jpg or jpeg is present.
    /// let rename = Rename::new([("image", "png;jpg;jpeg"), ("label", "cls")]);
    /// # let _ = rename;
    /// ```
    pub fn new<A: Into<String>, B: Into<String>>(renames: impl IntoIterator<Item = (A, B)>) -> Rename {
        Rename {
            renames: renames.into_iter().map(|(to, from)| (to.into(), from.into())).collect(),
            keep: true,
            handler: reraise_exception(),
        }
    }

    /// Drop the fields that were not renamed.
    pub fn only_renamed(mut self) -> Rename {
        self.keep = false;
        self
    }

    /// Decide what happens when a source field is missing.
    pub fn with_handler(mut self, handler: HandlerRef) -> Rename {
        self.handler = handler;
        self
    }
}

impl Stage for Rename {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let renames = self.renames.clone();
        let keep = self.keep;
        Box::new(
            input.map_sample_with(move |sample| rename_fields(&sample, &renames, keep).map(Some), self.handler.clone()),
        )
    }
}

/// Renames fields by glob pattern.
#[derive(Debug)]
pub struct RenameKeys {
    renames: Vec<(glob::Pattern, String)>,
    keep_unselected: bool,
    must_match: bool,
    duplicate_is_error: bool,
    handler: HandlerRef,
}

impl RenameKeys {
    /// Rename fields matching each pattern to the paired name.
    pub fn new<A: Into<String>>(renames: impl IntoIterator<Item = (A, &'static str)>) -> Result<RenameKeys> {
        let renames = renames
            .into_iter()
            .map(|(to, pattern)| {
                glob::Pattern::new(pattern)
                    .map(|p| (p, to.into()))
                    .map_err(|e| Error::value(format!("bad pattern {pattern:?}: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RenameKeys {
            renames,
            keep_unselected: false,
            must_match: true,
            duplicate_is_error: true,
            handler: reraise_exception(),
        })
    }

    /// Keep fields that matched no pattern.
    pub fn keep_unselected(mut self, keep: bool) -> RenameKeys {
        self.keep_unselected = keep;
        self
    }

    /// Whether every pattern must match something.
    pub fn must_match(mut self, must: bool) -> RenameKeys {
        self.must_match = must;
        self
    }

    /// Whether two fields renaming to the same name is an error.
    pub fn duplicate_is_error(mut self, is_error: bool) -> RenameKeys {
        self.duplicate_is_error = is_error;
        self
    }

    /// Decide what happens when renaming fails.
    pub fn with_handler(mut self, handler: HandlerRef) -> RenameKeys {
        self.handler = handler;
        self
    }
}

impl Stage for RenameKeys {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let renames = self.renames.clone();
        let (keep, must, dup) = (self.keep_unselected, self.must_match, self.duplicate_is_error);
        Box::new(input.map_sample_with(
            move |sample| rename_keys_in(&sample, &renames, keep, must, dup).map(Some),
            self.handler.clone(),
        ))
    }
}

/// Looks up the extra fields [`Associate`] adds to a sample.
pub type LookupFn = Arc<dyn Fn(&str) -> Vec<(String, Value)> + Send + Sync>;

/// Attaches extra fields to each sample, looked up by key.
pub struct Associate {
    lookup: LookupFn,
}

impl std::fmt::Debug for Associate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Associate")
    }
}

impl Associate {
    /// Add the fields `lookup` returns for each sample's key.
    pub fn new(lookup: impl Fn(&str) -> Vec<(String, Value)> + Send + Sync + 'static) -> Associate {
        Associate { lookup: Arc::new(lookup) }
    }

    /// Add fields from an in-memory table.
    pub fn from_table(table: std::collections::HashMap<String, Vec<(String, Value)>>) -> Associate {
        Associate::new(move |key| table.get(key).cloned().unwrap_or_default())
    }
}

impl Stage for Associate {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let lookup = self.lookup.clone();
        Box::new(input.map_sample(move |mut sample| {
            let key = sample.key().unwrap_or("").to_string();
            for (name, value) in lookup(&key) {
                sample.insert(name, value);
            }
            Ok(Some(sample))
        }))
    }
}

/// Keeps each sample with a fixed probability.
#[derive(Debug)]
pub struct RandomSubsample {
    probability: f64,
    seed: Option<u64>,
}

impl RandomSubsample {
    /// Keep each sample with probability `p`.
    pub fn new(p: f64) -> RandomSubsample {
        RandomSubsample { probability: p, seed: None }
    }

    /// Make the choice reproducible.
    pub fn with_seed(mut self, seed: u64) -> RandomSubsample {
        self.seed = Some(seed);
        self
    }
}

impl Stage for RandomSubsample {
    fn apply(&self, input: SampleStream) -> SampleStream {
        Box::new(input.rsample(self.probability, self.seed))
    }
}

/// Takes a slice of the stream.
#[derive(Debug)]
pub struct Slice {
    start: usize,
    count: Option<usize>,
    step: usize,
}

impl Slice {
    /// Take `count` samples starting at `start`.
    pub fn new(start: usize, count: Option<usize>) -> Slice {
        Slice { start, count, step: 1 }
    }

    /// Take every `step`th sample.
    pub fn with_step(mut self, step: usize) -> Slice {
        self.step = step.max(1);
        self
    }
}

impl Stage for Slice {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let stepped = input.skip(self.start).step_by(self.step);
        match self.count {
            Some(n) => Box::new(stepped.take(n)),
            None => Box::new(stepped),
        }
    }
}

/// Fails if the pipeline produced no samples at all.
///
/// The usual cause is having fewer shards than workers, which silently starves
/// some workers; failing is far easier to diagnose than an empty epoch.
#[derive(Debug, Clone)]
pub struct CheckEmpty {
    message: String,
}

impl Default for CheckEmpty {
    fn default() -> CheckEmpty {
        CheckEmpty {
            message: "no samples found; you may have fewer shards than workers. \
                      Disable this check with empty_check(false)."
                .to_string(),
        }
    }
}

impl CheckEmpty {
    /// Fail with this message instead of the default one.
    pub fn with_message(message: impl Into<String>) -> CheckEmpty {
        CheckEmpty { message: message.into() }
    }
}

impl Stage for CheckEmpty {
    fn apply(&self, input: SampleStream) -> SampleStream {
        Box::new(input.non_empty(self.message.clone()))
    }
}

/// Caches the whole stream in memory the first time it runs.
///
/// Only sensible for datasets that fit in memory; later epochs replay the cache
/// instead of touching the archives again.
///
/// The cache is filled lazily and only committed once the first pass reaches
/// the end, so abandoning an epoch part way through leaves the cache empty
/// rather than truncated.
#[derive(Debug, Default)]
pub struct MemoryCache {
    cached: Arc<std::sync::OnceLock<Vec<Sample>>>,
}

impl MemoryCache {
    /// An empty cache.
    pub fn new() -> MemoryCache {
        MemoryCache::default()
    }

    /// How many samples are cached, once the first pass has finished.
    pub fn len(&self) -> Option<usize> {
        self.cached.get().map(Vec::len)
    }

    /// Whether the cache is populated and empty.
    pub fn is_empty(&self) -> bool {
        self.cached.get().is_some_and(Vec::is_empty)
    }
}

impl Stage for MemoryCache {
    fn apply(&self, input: SampleStream) -> SampleStream {
        if let Some(cached) = self.cached.get() {
            return Box::new(cached.clone().into_iter().map(Ok));
        }
        Box::new(Filling { source: input, collected: Vec::new(), cache: self.cached.clone() })
    }
}

/// Passes samples through while collecting them for [`MemoryCache`].
struct Filling {
    source: SampleStream,
    collected: Vec<Sample>,
    cache: Arc<std::sync::OnceLock<Vec<Sample>>>,
}

impl Iterator for Filling {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        match self.source.next() {
            Some(Ok(sample)) => {
                self.collected.push(sample.clone());
                Some(Ok(sample))
            }
            Some(Err(e)) => Some(Err(e)),
            None => {
                // Only a complete pass is worth caching.
                let _ = self.cache.set(std::mem::take(&mut self.collected));
                None
            }
        }
    }
}

/// Logs samples as they pass, for debugging a pipeline.
#[derive(Debug)]
pub struct Trace {
    label: String,
    first: usize,
    every: Option<usize>,
    seen: AtomicUsize,
}

impl Trace {
    /// Log the first few samples under `label`.
    pub fn new(label: impl Into<String>) -> Trace {
        Trace { label: label.into(), first: 3, every: None, seen: AtomicUsize::new(0) }
    }

    /// Log this many samples at the start of each epoch.
    pub fn first(mut self, first: usize) -> Trace {
        self.first = first;
        self
    }

    /// Also log every `every`th sample.
    pub fn every(mut self, every: usize) -> Trace {
        self.every = Some(every.max(1));
        self
    }
}

impl Stage for Trace {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let label = self.label.clone();
        let (first, every) = (self.first, self.every);
        self.seen.store(0, Ordering::Relaxed);
        let mut seen = 0usize;
        Box::new(input.inspect(move |item| {
            seen += 1;
            let interesting = seen <= first || every.is_some_and(|n| seen % n == 0);
            if !interesting {
                return;
            }
            match item {
                Ok(sample) => {
                    log::info!("{label} [{seen}] {} {:?}", sample.key().unwrap_or("<no key>"), sample.field_names())
                }
                Err(e) => log::info!("{label} [{seen}] error: {e}"),
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{DataPipeline, Samples};

    fn samples(n: i64) -> Samples {
        Samples::new((0..n).map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Bytes(i.to_string().into()));
            s
        }))
    }

    #[test]
    fn shuffle_varies_between_epochs_when_deterministic() {
        let pipeline = DataPipeline::new().with(samples(50)).with(Shuffle::new(10).deterministic(7));
        let first: Vec<_> = pipeline.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();
        let second: Vec<_> = pipeline.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect();
        assert_ne!(first, second, "a deterministic shuffle still advances per epoch");
        assert_eq!(first.len(), 50);
    }

    #[test]
    fn shuffle_repeats_with_a_fixed_seed() {
        let a = DataPipeline::new().with(samples(20)).with(Shuffle::new(5).with_seed(3));
        let b = DataPipeline::new().with(samples(20)).with(Shuffle::new(5).with_seed(3));
        let keys =
            |p: &DataPipeline| -> Vec<String> { p.iter().map(|s| s.unwrap().key().unwrap().to_string()).collect() };
        assert_eq!(keys(&a), keys(&b));
    }

    #[test]
    fn decode_turns_bytes_into_values() {
        let pipeline = DataPipeline::new().with(samples(3)).with(Decode::basic());
        let decoded: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
        assert_eq!(decoded[2].get("cls").unwrap().as_i64(), Some(2));
    }

    #[test]
    fn map_and_select_compose() {
        let pipeline = DataPipeline::new()
            .with(samples(10))
            .with(Decode::basic())
            .with(SelectStage::new(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(0) % 2 == 0))
            .with(MapStage::new(|mut s: Sample| {
                let cls = s.get("cls").and_then(Value::as_i64).unwrap_or(0);
                s.insert("cls", Value::Int(cls * 10));
                Ok(Some(s))
            }));
        let values: Vec<i64> = pipeline.iter().map(|s| s.unwrap().get("cls").unwrap().as_i64().unwrap()).collect();
        assert_eq!(values, [0, 20, 40, 60, 80]);
    }

    #[test]
    fn rename_resolves_alternatives() {
        let pipeline = DataPipeline::new().with(samples(1)).with(Rename::new([("label", "png;cls")]));
        let renamed = pipeline.iter().next().unwrap().unwrap();
        assert!(renamed.contains_key("label"));
        assert!(!renamed.contains_key("cls"));
    }

    #[test]
    fn rename_keys_matches_by_pattern() {
        let stage = RenameKeys::new([("label", "cl?")]).unwrap();
        let pipeline = DataPipeline::new().with(samples(1)).with(stage);
        assert_eq!(pipeline.iter().next().unwrap().unwrap().field_names(), ["label"]);
    }

    #[test]
    fn associate_adds_fields_by_key() {
        let mut table = std::collections::HashMap::new();
        table.insert("k1".to_string(), vec![("extra".to_string(), Value::Int(99))]);

        let pipeline = DataPipeline::new().with(samples(3)).with(Associate::from_table(table));
        let samples: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
        assert_eq!(samples[1].get("extra").unwrap().as_i64(), Some(99));
        assert!(!samples[0].contains_key("extra"));
    }

    #[test]
    fn slice_and_subsample_reduce_the_stream() {
        let sliced = DataPipeline::new().with(samples(10)).with(Slice::new(2, Some(3)));
        assert_eq!(sliced.iter().count(), 3);

        let stepped = DataPipeline::new().with(samples(10)).with(Slice::new(0, None).with_step(3));
        assert_eq!(stepped.iter().count(), 4);

        let sampled = DataPipeline::new().with(samples(100)).with(RandomSubsample::new(0.0).with_seed(1));
        assert_eq!(sampled.iter().count(), 0);
    }

    #[test]
    fn check_empty_reports_a_starved_pipeline() {
        let empty = DataPipeline::new().with(Samples::new([])).with(CheckEmpty::default());
        let outcome: Vec<_> = empty.iter().collect();
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]));

        let full = DataPipeline::new().with(samples(1)).with(CheckEmpty::default());
        assert!(full.iter().all(|s| s.is_ok()));
    }

    #[test]
    fn memory_cache_replays_the_first_pass() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counting = counter.clone();

        let pipeline = DataPipeline::new()
            .with(samples(5))
            .with(MapStage::new(move |s| {
                counting.fetch_add(1, Ordering::Relaxed);
                Ok(Some(s))
            }))
            .with(MemoryCache::new());

        assert_eq!(pipeline.iter().count(), 5);
        assert_eq!(counter.load(Ordering::Relaxed), 5);

        assert_eq!(pipeline.iter().count(), 5);
        assert_eq!(counter.load(Ordering::Relaxed), 5, "the second pass must come from the cache");
    }

    #[test]
    fn memory_cache_ignores_an_abandoned_first_pass() {
        let cache = Arc::new(MemoryCache::new());
        let pipeline = DataPipeline::new().with(samples(10)).with(cache.clone());

        assert_eq!(pipeline.iter().take(3).count(), 3);
        assert_eq!(cache.len(), None, "a partial pass must not be cached");

        assert_eq!(pipeline.iter().count(), 10);
        assert_eq!(cache.len(), Some(10));
    }

    #[test]
    fn trace_passes_samples_through_unchanged() {
        let pipeline = DataPipeline::new().with(samples(4)).with(Trace::new("test").first(2));
        assert_eq!(pipeline.iter().count(), 4);
    }
}
