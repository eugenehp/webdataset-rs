//! Iterator adapters for sample streams.
//!
//! Every filter here is a plain [`Iterator`] adapter, so it composes with the
//! standard library and with user code. The [`SampleIteratorExt`] and
//! [`TupleIteratorExt`] traits add them as methods:
//!
//! ```
//! use webdataset::filters::SampleIteratorExt;
//! use webdataset_core::{Sample, Value};
//!
//! let samples = (0..10).map(|i| {
//!     let mut s = Sample::with_key(format!("k{i}"));
//!     s.insert("cls", Value::Int(i));
//!     Ok(s)
//! });
//!
//! let batches: Vec<_> = samples
//!     .select(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(0) % 2 == 0)
//!     .batched(2, true)
//!     .collect::<Result<Vec<_>, _>>()?;
//!
//! assert_eq!(batches.len(), 3, "five even samples in batches of two");
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! ## Errors
//!
//! An adapter passes errors from upstream through untouched — the stage that
//! produced them has already consulted its own handler. Errors an adapter
//! raises itself go through the [`Handler`](webdataset_core::Handler) it was
//! given, which by default forwards them downstream.

use std::sync::Arc;

use rand::prelude::*;
use rand::rngs::StdRng;
use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{Action, HandlerRef, reraise_exception};
use webdataset_core::sample::{KEY, Sample, is_meta};
use webdataset_core::value::Value;

use crate::batch::{collate_samples, collate_tuples};

/// Seed an RNG from the operating system, for when no seed was configured.
fn entropy_seed() -> u64 {
    let mut seed = [0u8; 8];
    rand::rng().fill_bytes(&mut seed);
    u64::from_ne_bytes(seed)
}

/// Take a random element out of `buffer`, filling the hole with the last one.
fn pick(buffer: &mut Vec<Sample>, rng: &mut StdRng) -> Option<Sample> {
    if buffer.is_empty() {
        return None;
    }
    let index = rng.random_range(0..buffer.len());
    Some(buffer.swap_remove(index))
}

/// Shuffles a stream through a fixed-size buffer.
///
/// Samples arrive in shard order, which correlates strongly with label order in
/// most datasets, so training needs them shuffled. Reading the whole dataset
/// into memory is not an option, so a buffer of `bufsize` samples is kept and a
/// random one is emitted each time a new one arrives. Output begins once
/// `initial` samples are buffered, which trades a little randomness at the
/// start of an epoch for a much shorter warm-up.
pub struct Shuffle<I> {
    source: I,
    buffer: Vec<Sample>,
    bufsize: usize,
    initial: usize,
    rng: StdRng,
    drained: bool,
}

impl<I> Shuffle<I> {
    /// Shuffle through a buffer of `bufsize`, emitting once `initial` are held.
    pub fn new(source: I, bufsize: usize, initial: usize, seed: Option<u64>) -> Shuffle<I> {
        let bufsize = bufsize.max(1);
        Shuffle {
            source,
            buffer: Vec::with_capacity(bufsize.min(4096)),
            bufsize,
            initial: initial.clamp(1, bufsize),
            rng: StdRng::seed_from_u64(seed.unwrap_or_else(entropy_seed)),
            drained: false,
        }
    }
}

impl<I: Iterator<Item = Result<Sample>>> Iterator for Shuffle<I> {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        while !self.drained {
            match self.source.next() {
                Some(Ok(sample)) => {
                    self.buffer.push(sample);
                    // Pull a second sample while the buffer is still filling so
                    // that the warm-up ends in half the samples.
                    if self.buffer.len() < self.bufsize {
                        match self.source.next() {
                            Some(Ok(extra)) => self.buffer.push(extra),
                            Some(Err(e)) => return Some(Err(e)),
                            None => self.drained = true,
                        }
                    }
                    if self.buffer.len() >= self.initial {
                        return pick(&mut self.buffer, &mut self.rng).map(Ok);
                    }
                }
                Some(Err(e)) => return Some(Err(e)),
                None => self.drained = true,
            }
        }
        pick(&mut self.buffer, &mut self.rng).map(Ok)
    }
}

/// Keeps the samples a predicate accepts.
pub struct Select<I, P> {
    source: I,
    predicate: P,
}

impl<I, P> Iterator for Select<I, P>
where
    I: Iterator<Item = Result<Sample>>,
    P: FnMut(&Sample) -> bool,
{
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        loop {
            match self.source.next()? {
                Ok(sample) if (self.predicate)(&sample) => return Some(Ok(sample)),
                Ok(_) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Applies a function to each sample, dropping the ones it maps to `None`.
pub struct MapSamples<I, F> {
    source: I,
    f: F,
    handler: HandlerRef,
    done: bool,
}

impl<I, F> Iterator for MapSamples<I, F>
where
    I: Iterator<Item = Result<Sample>>,
    F: FnMut(Sample) -> Result<Option<Sample>>,
{
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        loop {
            if self.done {
                return None;
            }
            let sample = match self.source.next()? {
                Ok(sample) => sample,
                Err(e) => return Some(Err(e)),
            };
            // The key survives the mapping, so downstream stages can still
            // identify the sample even if the function rebuilt it.
            let key = sample.key().map(str::to_string);
            match (self.f)(sample) {
                Ok(Some(mut mapped)) => {
                    if let Some(key) = key {
                        if !mapped.contains_key(KEY) {
                            mapped.set_key(key);
                        }
                    }
                    return Some(Ok(mapped));
                }
                Ok(None) => continue,
                Err(e) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        return None;
                    }
                    Action::Reraise => return Some(Err(e)),
                },
            }
        }
    }
}

/// Picks fields out of each sample, producing tuples.
pub struct ToTuple<I> {
    source: I,
    specs: Vec<String>,
    handler: HandlerRef,
    missing_is_error: bool,
    done: bool,
}

impl<I: Iterator<Item = Result<Sample>>> Iterator for ToTuple<I> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Result<Vec<Value>>> {
        loop {
            if self.done {
                return None;
            }
            let sample = match self.source.next()? {
                Ok(sample) => sample,
                Err(e) => return Some(Err(e)),
            };
            let mut row = Vec::with_capacity(self.specs.len());
            let mut failure = None;
            for spec in &self.specs {
                match sample.get_first_spec(spec) {
                    Some(value) => row.push(value.clone()),
                    None if self.missing_is_error => {
                        failure = Some(Error::MissingKey {
                            wanted: spec.split(';').map(str::to_string).collect(),
                            available: sample.keys().map(str::to_string).collect(),
                        });
                        break;
                    }
                    None => row.push(Value::Null),
                }
            }
            match failure {
                None => return Some(Ok(row)),
                Some(e) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        return None;
                    }
                    Action::Reraise => return Some(Err(e)),
                },
            }
        }
    }
}

/// Picks fields by glob pattern, producing tuples.
pub struct ExtractKeys<I> {
    source: I,
    patterns: Vec<Vec<glob::Pattern>>,
    duplicate_is_error: bool,
    ignore_missing: bool,
    handler: HandlerRef,
    done: bool,
}

impl<I: Iterator<Item = Result<Sample>>> Iterator for ExtractKeys<I> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Result<Vec<Value>>> {
        loop {
            if self.done {
                return None;
            }
            let sample = match self.source.next()? {
                Ok(sample) => sample,
                Err(e) => return Some(Err(e)),
            };
            match extract(&sample, &self.patterns, self.duplicate_is_error, self.ignore_missing) {
                Ok(row) => return Some(Ok(row)),
                Err(e) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        return None;
                    }
                    Action::Reraise => return Some(Err(e)),
                },
            }
        }
    }
}

fn extract(
    sample: &Sample,
    patterns: &[Vec<glob::Pattern>],
    duplicate_is_error: bool,
    ignore_missing: bool,
) -> Result<Vec<Value>> {
    let mut row = Vec::with_capacity(patterns.len());
    for alternatives in patterns {
        let matches: Vec<&str> = sample
            .keys()
            .filter(|name| alternatives.iter().any(|p| p.matches(name) || p.matches(&format!(".{name}"))))
            .collect();
        match matches.len() {
            0 if ignore_missing => continue,
            0 => {
                return Err(Error::MissingKey {
                    wanted: alternatives.iter().map(|p| p.as_str().to_string()).collect(),
                    available: sample.keys().map(str::to_string).collect(),
                });
            }
            n if n > 1 && duplicate_is_error => {
                return Err(Error::value(format!(
                    "{:?} all match {:?}",
                    matches,
                    alternatives.iter().map(glob::Pattern::as_str).collect::<Vec<_>>()
                )));
            }
            _ => row.push(sample.get(matches[0]).expect("name came from the sample").clone()),
        }
    }
    Ok(row)
}

/// Collects items into fixed-size groups.
pub struct Listed<I: Iterator> {
    source: I,
    size: usize,
    partial: bool,
}

impl<I, T> Iterator for Listed<I>
where
    I: Iterator<Item = Result<T>>,
{
    type Item = Result<Vec<T>>;

    fn next(&mut self) -> Option<Result<Vec<T>>> {
        let mut group = Vec::with_capacity(self.size);
        while group.len() < self.size {
            match self.source.next() {
                Some(Ok(item)) => group.push(item),
                Some(Err(e)) => return Some(Err(e)),
                None => break,
            }
        }
        match group.len() {
            0 => None,
            n if n < self.size && !self.partial => None,
            _ => Some(Ok(group)),
        }
    }
}

/// Keeps each item with probability `p`.
pub struct RandomSubsample<I> {
    source: I,
    probability: f64,
    rng: StdRng,
}

impl<I, T> Iterator for RandomSubsample<I>
where
    I: Iterator<Item = Result<T>>,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Result<T>> {
        loop {
            match self.source.next()? {
                Ok(item) if self.rng.random::<f64>() < self.probability => return Some(Ok(item)),
                Ok(_) => continue,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Splits collated batches back into samples.
pub struct Unbatched<I> {
    source: I,
    pending: std::vec::IntoIter<Sample>,
}

impl<I: Iterator<Item = Result<Sample>>> Iterator for Unbatched<I> {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        loop {
            if let Some(sample) = self.pending.next() {
                return Some(Ok(sample));
            }
            match self.source.next()? {
                Ok(batch) => match crate::batch::uncollate_sample(&batch) {
                    Ok(samples) => self.pending = samples.into_iter(),
                    Err(e) => return Some(Err(e)),
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Fails if the stream turns out to be empty.
pub struct NonEmpty<I> {
    source: I,
    seen: usize,
    message: String,
    reported: bool,
}

impl<I, T> Iterator for NonEmpty<I>
where
    I: Iterator<Item = Result<T>>,
{
    type Item = Result<T>;

    fn next(&mut self) -> Option<Result<T>> {
        match self.source.next() {
            Some(item) => {
                self.seen += 1;
                Some(item)
            }
            None if self.seen == 0 && !self.reported => {
                self.reported = true;
                Some(Err(Error::Empty(self.message.clone())))
            }
            None => None,
        }
    }
}

/// Rename fields, resolving each target from a `;`-separated alternation.
///
/// With `keep` set, fields that were not consumed by a renaming are carried
/// over; otherwise only the renamed fields survive. Metadata is always kept.
pub fn rename_fields(sample: &Sample, renames: &[(String, String)], keep: bool) -> Result<Sample> {
    let mut out = Sample::new();
    if keep {
        let consumed: Vec<&str> = renames.iter().flat_map(|(_, from)| from.split(';')).collect();
        for (name, value) in sample {
            if !consumed.contains(&name.as_str()) {
                out.insert(name.clone(), value.clone());
            }
        }
    } else {
        for (name, value) in sample {
            if is_meta(name) {
                out.insert(name.clone(), value.clone());
            }
        }
    }
    for (to, from) in renames {
        out.insert(to.clone(), sample.require_first_spec(from)?.clone());
    }
    Ok(out)
}

/// Rename fields by glob pattern, matching the way files are named.
///
/// Patterns are matched against the last path component of each field name,
/// lowercased. Later patterns win, matching the Python implementation.
pub fn rename_keys_in(
    sample: &Sample,
    renames: &[(glob::Pattern, String)],
    keep_unselected: bool,
    must_match: bool,
    duplicate_is_error: bool,
) -> Result<Sample> {
    let mut out = Sample::new();
    let mut matched = vec![false; renames.len()];

    for (path, value) in sample {
        let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
        let hit = renames.iter().enumerate().rev().find(|(_, (pattern, _))| pattern.matches(&name));
        match hit {
            Some((index, (_, target))) => {
                matched[index] = true;
                if out.contains_key(target) {
                    if duplicate_is_error {
                        return Err(Error::value(format!("{path}: {target} is already set after renaming")));
                    }
                    continue;
                }
                out.insert(target.clone(), value.clone());
            }
            None if keep_unselected => {
                out.insert(path.clone(), value.clone());
            }
            None => {}
        }
    }

    if must_match && !matched.iter().all(|m| *m) {
        let missed: Vec<&str> =
            renames.iter().zip(&matched).filter(|(_, hit)| !**hit).map(|((pattern, _), _)| pattern.as_str()).collect();
        return Err(Error::value(format!("patterns {missed:?} matched nothing in {:?}", sample.field_names())));
    }
    Ok(out)
}

/// Adapters for streams of samples.
pub trait SampleIteratorExt: Iterator<Item = Result<Sample>> + Sized {
    /// Shuffle through a buffer of `bufsize` samples.
    fn shuffled(self, bufsize: usize, seed: Option<u64>) -> Shuffle<Self> {
        Shuffle::new(self, bufsize, bufsize.div_ceil(10).max(1), seed)
    }

    /// Keep the samples `predicate` accepts.
    fn select<P: FnMut(&Sample) -> bool>(self, predicate: P) -> Select<Self, P> {
        Select { source: self, predicate }
    }

    /// Apply `f` to each sample, dropping those it maps to `None`.
    fn map_sample<F: FnMut(Sample) -> Result<Option<Sample>>>(self, f: F) -> MapSamples<Self, F> {
        MapSamples { source: self, f, handler: reraise_exception(), done: false }
    }

    /// Apply `f`, sending failures to `handler`.
    fn map_sample_with<F: FnMut(Sample) -> Result<Option<Sample>>>(
        self,
        f: F,
        handler: HandlerRef,
    ) -> MapSamples<Self, F> {
        MapSamples { source: self, f, handler, done: false }
    }

    /// Project each sample onto the named fields.
    ///
    /// Each spec may list alternatives, as in `"png;jpg;jpeg"`.
    fn to_tuple<S: AsRef<str>>(self, specs: impl IntoIterator<Item = S>) -> ToTuple<Self> {
        ToTuple {
            source: self,
            specs: specs.into_iter().map(|s| s.as_ref().to_string()).collect(),
            handler: reraise_exception(),
            missing_is_error: true,
            done: false,
        }
    }

    /// Project each sample onto fields chosen by glob pattern.
    fn extract_keys<S: AsRef<str>>(self, patterns: impl IntoIterator<Item = S>) -> Result<ExtractKeys<Self>> {
        let patterns = patterns
            .into_iter()
            .map(|spec| {
                spec.as_ref()
                    .split(';')
                    .map(|p| glob::Pattern::new(p).map_err(|e| Error::value(format!("bad pattern {p:?}: {e}"))))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ExtractKeys {
            source: self,
            patterns,
            duplicate_is_error: true,
            ignore_missing: false,
            handler: reraise_exception(),
            done: false,
        })
    }

    /// Group samples into lists of `size`, without collating them.
    fn listed(self, size: usize, partial: bool) -> Listed<Self> {
        Listed { source: self, size: size.max(1), partial }
    }

    /// Group and collate samples into batches of `size`.
    fn batched(self, size: usize, partial: bool) -> Batched<Self> {
        Batched { inner: self.listed(size, partial) }
    }

    /// Split collated batches back into samples.
    fn unbatched(self) -> Unbatched<Self> {
        Unbatched { source: self, pending: Vec::new().into_iter() }
    }

    /// Keep each sample with probability `p`.
    fn rsample(self, p: f64, seed: Option<u64>) -> RandomSubsample<Self> {
        RandomSubsample {
            source: self,
            probability: p.clamp(0.0, 1.0),
            rng: StdRng::seed_from_u64(seed.unwrap_or_else(entropy_seed)),
        }
    }

    /// Fail with [`Error::Empty`] if the stream produced nothing.
    fn non_empty(self, message: impl Into<String>) -> NonEmpty<Self> {
        NonEmpty { source: self, seen: 0, message: message.into(), reported: false }
    }

    /// Take `count` samples starting at `start`, stepping by `step`.
    fn sliced(self, start: usize, count: Option<usize>, step: usize) -> Box<dyn Iterator<Item = Result<Sample>>>
    where
        Self: 'static,
    {
        let stepped = self.skip(start).step_by(step.max(1));
        match count {
            Some(n) => Box::new(stepped.take(n)),
            None => Box::new(stepped),
        }
    }
}

impl<I: Iterator<Item = Result<Sample>>> SampleIteratorExt for I {}

/// Collated batches of samples.
pub struct Batched<I: Iterator> {
    inner: Listed<I>,
}

impl<I: Iterator<Item = Result<Sample>>> Iterator for Batched<I> {
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        match self.inner.next()? {
            Ok(group) => Some(collate_samples(group)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// A function applied to one position of a tuple by
/// [`map_tuple`](TupleIteratorExt::map_tuple).
pub type ValueFn = Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>;

/// Adapters for streams of tuples, as produced by
/// [`to_tuple`](SampleIteratorExt::to_tuple).
pub trait TupleIteratorExt: Iterator<Item = Result<Vec<Value>>> + Sized {
    /// Apply one function per tuple position; `None` leaves that position alone.
    fn map_tuple(self, fs: Vec<Option<ValueFn>>) -> MapTuple<Self> {
        MapTuple { source: self, fs, handler: reraise_exception(), done: false }
    }

    /// Group tuples into lists of `size`, without collating them.
    fn listed(self, size: usize, partial: bool) -> Listed<Self> {
        Listed { source: self, size: size.max(1), partial }
    }

    /// Group and collate tuples into batches of `size`.
    fn batched(self, size: usize, partial: bool) -> BatchedTuples<Self> {
        BatchedTuples { inner: TupleIteratorExt::listed(self, size, partial) }
    }
}

impl<I: Iterator<Item = Result<Vec<Value>>>> TupleIteratorExt for I {}

/// Collated batches of tuples.
pub struct BatchedTuples<I: Iterator> {
    inner: Listed<I>,
}

impl<I: Iterator<Item = Result<Vec<Value>>>> Iterator for BatchedTuples<I> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Result<Vec<Value>>> {
        match self.inner.next()? {
            Ok(rows) => Some(collate_tuples(rows)),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Applies a function to each position of a tuple.
pub struct MapTuple<I> {
    source: I,
    fs: Vec<Option<ValueFn>>,
    handler: HandlerRef,
    done: bool,
}

impl<I: Iterator<Item = Result<Vec<Value>>>> Iterator for MapTuple<I> {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Result<Vec<Value>>> {
        loop {
            if self.done {
                return None;
            }
            let mut row = match self.source.next()? {
                Ok(row) => row,
                Err(e) => return Some(Err(e)),
            };
            let mut failure = None;
            for (i, f) in self.fs.iter().enumerate().take(row.len()) {
                let Some(f) = f else { continue };
                match f(row[i].clone()) {
                    Ok(value) => row[i] = value,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
            match failure {
                None => return Some(Ok(row)),
                Some(e) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        return None;
                    }
                    Action::Reraise => return Some(Err(e)),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(n: i64) -> impl Iterator<Item = Result<Sample>> {
        (0..n).map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s.insert("txt", Value::Text(format!("t{i}")));
            Ok(s)
        })
    }

    fn keys(stream: impl Iterator<Item = Result<Sample>>) -> Vec<String> {
        stream.map(|s| s.unwrap().key().unwrap().to_string()).collect()
    }

    #[test]
    fn shuffle_keeps_every_sample() {
        let shuffled = keys(samples(100).shuffled(10, Some(7)));
        assert_eq!(shuffled.len(), 100);

        let mut sorted = shuffled.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 100, "no duplicates or losses");
        assert_ne!(shuffled, keys(samples(100)), "the order should actually change");
    }

    #[test]
    fn shuffle_is_reproducible_for_a_given_seed() {
        assert_eq!(keys(samples(50).shuffled(10, Some(1))), keys(samples(50).shuffled(10, Some(1))));
        assert_ne!(keys(samples(50).shuffled(10, Some(1))), keys(samples(50).shuffled(10, Some(2))));
    }

    #[test]
    fn shuffle_handles_streams_smaller_than_the_buffer() {
        assert_eq!(keys(samples(3).shuffled(1000, Some(1))).len(), 3);
        assert_eq!(keys(samples(0).shuffled(10, Some(1))).len(), 0);
    }

    #[test]
    fn select_filters() {
        let kept = keys(samples(10).select(|s| s.get("cls").and_then(Value::as_i64).unwrap() < 3));
        assert_eq!(kept, ["k0", "k1", "k2"]);
    }

    #[test]
    fn map_drops_none_and_preserves_the_key() {
        let mapped: Vec<Sample> = samples(4)
            .map_sample(|s| {
                let cls = s.get("cls").and_then(Value::as_i64).unwrap();
                if cls % 2 == 1 {
                    return Ok(None);
                }
                let mut out = Sample::new();
                out.insert("doubled", Value::Int(cls * 2));
                Ok(Some(out))
            })
            .map(|s| s.unwrap())
            .collect();

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[1].key(), Some("k2"), "the key carries over");
        assert_eq!(mapped[1].get("doubled").unwrap().as_i64(), Some(4));
    }

    #[test]
    fn map_routes_failures_to_the_handler() {
        let failing = |_: Sample| -> Result<Option<Sample>> { Err(Error::value("nope")) };

        let reraised: Vec<_> = samples(3).map_sample(failing).collect();
        assert!(reraised.iter().all(Result::is_err));

        let skipped = samples(3).map_sample_with(failing, webdataset_core::handlers::ignore_and_continue()).count();
        assert_eq!(skipped, 0);

        let stopped = samples(3).map_sample_with(failing, webdataset_core::handlers::ignore_and_stop()).count();
        assert_eq!(stopped, 0);
    }

    #[test]
    fn to_tuple_projects_fields() {
        let rows: Vec<Vec<Value>> = samples(2).to_tuple(["cls", "txt"]).map(|r| r.unwrap()).collect();
        assert_eq!(rows[0], vec![Value::Int(0), Value::Text("t0".into())]);

        let missing: Vec<_> = samples(2).to_tuple(["nope"]).collect();
        assert!(missing[0].is_err());

        let alternative: Vec<Vec<Value>> = samples(1).to_tuple(["png;cls"]).map(|r| r.unwrap()).collect();
        assert_eq!(alternative[0], vec![Value::Int(0)]);
    }

    #[test]
    fn extract_keys_uses_glob_patterns() {
        let rows: Vec<Vec<Value>> =
            samples(1).extract_keys(["*.cls;cls", "txt"]).unwrap().map(|r| r.unwrap()).collect();
        assert_eq!(rows[0], vec![Value::Int(0), Value::Text("t0".into())]);
    }

    #[test]
    fn listed_and_batched_group_samples() {
        let lists: Vec<Vec<Sample>> = samples(5).listed(2, true).map(|b| b.unwrap()).collect();
        assert_eq!(lists.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 1]);

        let dropped: Vec<Vec<Sample>> = samples(5).listed(2, false).map(|b| b.unwrap()).collect();
        assert_eq!(dropped.len(), 2, "the partial batch is dropped");

        let batches: Vec<Sample> = samples(4).batched(2, true).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].get("cls").unwrap().as_tensor().unwrap().shape(), &[2]);
    }

    #[test]
    fn unbatched_reverses_batching() {
        let round_trip = keys(samples(6).batched(2, true).unbatched());
        assert_eq!(round_trip, ["k0", "k1", "k2", "k3", "k4", "k5"]);
    }

    #[test]
    fn rsample_keeps_roughly_the_requested_fraction() {
        assert_eq!(samples(100).rsample(1.0, Some(1)).count(), 100);
        assert_eq!(samples(100).rsample(0.0, Some(1)).count(), 0);
        let half = samples(1000).rsample(0.5, Some(42)).count();
        assert!((400..600).contains(&half), "kept {half} of 1000");
    }

    #[test]
    fn non_empty_reports_an_empty_stream() {
        assert_eq!(samples(3).non_empty("no data").filter(Result::is_ok).count(), 3);
        let outcome: Vec<_> = samples(0).non_empty("no data").collect();
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]));
    }

    #[test]
    fn sliced_skips_steps_and_truncates() {
        assert_eq!(keys(samples(10).sliced(2, Some(3), 1)), ["k2", "k3", "k4"]);
        assert_eq!(keys(samples(10).sliced(0, Some(3), 2)), ["k0", "k2", "k4"]);
    }

    #[test]
    fn rename_keeps_or_drops_the_rest() {
        let sample = samples(1).next().unwrap().unwrap();
        let renames = vec![("label".to_string(), "cls".to_string())];

        let kept = rename_fields(&sample, &renames, true).unwrap();
        assert_eq!(kept.get("label").unwrap().as_i64(), Some(0));
        assert!(kept.contains_key("txt"), "unrelated fields stay");
        assert!(!kept.contains_key("cls"), "the source field is consumed");

        let dropped = rename_fields(&sample, &renames, false).unwrap();
        assert_eq!(dropped.field_names(), ["label"]);
        assert_eq!(dropped.key(), Some("k0"), "metadata always survives");
    }

    #[test]
    fn rename_keys_matches_patterns() {
        let sample = samples(1).next().unwrap().unwrap();
        let renames = vec![(glob::Pattern::new("cls").unwrap(), "label".to_string())];

        let renamed = rename_keys_in(&sample, &renames, false, true, true).unwrap();
        assert_eq!(renamed.field_names(), ["label"]);

        let kept = rename_keys_in(&sample, &renames, true, true, true).unwrap();
        assert!(kept.contains_key("txt"));

        let unmatched = vec![(glob::Pattern::new("nope").unwrap(), "x".to_string())];
        assert!(rename_keys_in(&sample, &unmatched, false, true, true).is_err());
        assert!(rename_keys_in(&sample, &unmatched, false, false, true).is_ok());
    }

    #[test]
    fn map_tuple_applies_per_position() {
        let double: ValueFn = Arc::new(|v: Value| Ok(Value::Int(v.as_i64().unwrap_or(0) * 2)));
        let rows: Vec<Vec<Value>> =
            samples(2).to_tuple(["cls", "txt"]).map_tuple(vec![Some(double), None]).map(|r| r.unwrap()).collect();

        assert_eq!(rows[1][0], Value::Int(2));
        assert_eq!(rows[1][1], Value::Text("t1".into()));
    }

    #[test]
    fn tuple_batching_collates_columns() {
        let batches: Vec<Vec<Value>> = samples(4).to_tuple(["cls"]).batched(2, true).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0][0].as_tensor().unwrap().shape(), &[2]);
    }

    #[test]
    fn errors_from_upstream_pass_through_untouched() {
        let stream = vec![Ok(Sample::with_key("a")), Err(Error::value("upstream")), Ok(Sample::with_key("b"))];
        let outcome: Vec<_> = stream.into_iter().select(|_| true).collect();
        assert_eq!(outcome.len(), 3);
        assert!(outcome[1].is_err());
    }
}
