//! Stream adapters for sample streams.
//!
//! These mirror the iterator adapters in [`filters`](crate::filters) one for
//! one, and share their logic: shuffling uses the same buffer and the same
//! draw, batching the same collation, decoding the same decoder. Only the
//! plumbing differs, which is why a blocking and an asynchronous pipeline over
//! the same shards produce the same samples.
//!
//! ```
//! use futures_util::TryStreamExt;
//! use webdataset::asynch::AsyncSampleStreamExt;
//! use webdataset_core::{Sample, Value};
//!
//! # futures_executor::block_on(async {
//! let samples = futures_util::stream::iter((0..10).map(|i| {
//!     let mut s = Sample::with_key(format!("k{i}"));
//!     s.insert("cls", Value::Int(i));
//!     Ok(s)
//! }));
//!
//! let batches: Vec<_> = samples
//!     .select(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(0) % 2 == 0)
//!     .batched(2, true)
//!     .try_collect()
//!     .await?;
//!
//! assert_eq!(batches.len(), 3, "five even samples in batches of two");
//! # Ok::<(), webdataset_core::Error>(())
//! # }).unwrap();
//! ```
//!
//! ## Errors
//!
//! As in the blocking pipeline, an adapter passes errors from upstream through
//! untouched and sends only the errors it raises itself to its
//! [`Handler`](webdataset_core::Handler).

use std::pin::Pin;
use std::sync::Arc;

use futures_core::Stream;
use futures_util::stream::{self, StreamExt};
use rand::prelude::*;
use rand::rngs::StdRng;
use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{Action, HandlerRef, reraise_exception};
use webdataset_core::sample::{KEY, Sample};
use webdataset_core::value::Value;

use crate::batch::{collate_samples, collate_tuples};
use crate::decode::Decoder;

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

/// Adapters for streams of samples.
pub trait AsyncSampleStreamExt: Stream<Item = Result<Sample>> + Sized + Send + 'static {
    /// Shuffle through a buffer of `bufsize` samples.
    ///
    /// The buffer, the warm-up and the draw are the blocking implementation's;
    /// see [`filters::Shuffle`](crate::filters::Shuffle) for why it works this
    /// way.
    fn shuffled(self, bufsize: usize, seed: Option<u64>) -> BoxStream<Result<Sample>> {
        let bufsize = bufsize.max(1);
        let initial = bufsize.div_ceil(10).max(1);

        /// What the unfold carries between samples.
        struct State<S> {
            source: Pin<Box<S>>,
            buffer: Vec<Sample>,
            rng: StdRng,
            drained: bool,
        }

        let state = State {
            source: Box::pin(self),
            buffer: Vec::new(),
            rng: StdRng::seed_from_u64(seed.unwrap_or_else(entropy_seed)),
            drained: false,
        };

        Box::pin(stream::unfold(Some(state), move |carried| async move {
            let mut state = carried?;
            while !state.drained {
                match state.source.next().await {
                    Some(Ok(sample)) => {
                        state.buffer.push(sample);
                        // Pull a second sample while the buffer is still
                        // filling, so the warm-up ends in half the samples.
                        if state.buffer.len() < bufsize {
                            match state.source.next().await {
                                Some(Ok(extra)) => state.buffer.push(extra),
                                Some(Err(e)) => return Some((Err(e), Some(state))),
                                None => state.drained = true,
                            }
                        }
                        if state.buffer.len() >= initial {
                            let picked = pick(&mut state.buffer, &mut state.rng).expect("the buffer is not empty");
                            return Some((Ok(picked), Some(state)));
                        }
                    }
                    Some(Err(e)) => return Some((Err(e), Some(state))),
                    None => state.drained = true,
                }
            }
            let picked = pick(&mut state.buffer, &mut state.rng)?;
            Some((Ok(picked), Some(state)))
        }))
    }

    /// Keep the samples `predicate` accepts.
    fn select<P>(self, predicate: P) -> BoxStream<Result<Sample>>
    where
        P: FnMut(&Sample) -> bool + Send + 'static,
    {
        let mut predicate = predicate;
        Box::pin(self.filter(move |item| {
            let keep = match item {
                Ok(sample) => predicate(sample),
                Err(_) => true,
            };
            async move { keep }
        }))
    }

    /// Apply `f` to each sample, dropping those it maps to `None`.
    fn map_sample<F>(self, f: F) -> BoxStream<Result<Sample>>
    where
        F: FnMut(Sample) -> Result<Option<Sample>> + Send + 'static,
    {
        self.map_sample_with(f, reraise_exception())
    }

    /// Apply `f`, sending failures to `handler`.
    fn map_sample_with<F>(self, f: F, handler: HandlerRef) -> BoxStream<Result<Sample>>
    where
        F: FnMut(Sample) -> Result<Option<Sample>> + Send + 'static,
    {
        /// What the unfold carries between samples.
        struct State<S, F> {
            source: Pin<Box<S>>,
            f: F,
            handler: HandlerRef,
        }

        let state = State { source: Box::pin(self), f, handler };

        Box::pin(stream::unfold(Some(state), |carried| async move {
            let mut state = carried?;
            loop {
                let sample = match state.source.next().await? {
                    Ok(sample) => sample,
                    Err(e) => return Some((Err(e), Some(state))),
                };
                // The key survives the mapping, so downstream stages can still
                // identify the sample even if the function rebuilt it.
                let key = sample.key().map(str::to_string);
                match (state.f)(sample) {
                    Ok(Some(mut mapped)) => {
                        if let Some(key) = key {
                            if !mapped.contains_key(KEY) {
                                mapped.set_key(key);
                            }
                        }
                        return Some((Ok(mapped), Some(state)));
                    }
                    Ok(None) => continue,
                    Err(e) => match state.handler.handle(&e) {
                        Action::Continue => continue,
                        Action::Stop => return None,
                        Action::Reraise => return Some((Err(e), Some(state))),
                    },
                }
            }
        }))
    }

    /// Decode every field of every sample.
    fn decode(self, decoder: Arc<Decoder>) -> BoxStream<Result<Sample>> {
        self.map_sample(move |sample| decoder.decode(sample).map(Some))
    }

    /// Decode with the default handler chain.
    fn decode_basic(self) -> BoxStream<Result<Sample>> {
        self.decode(Arc::new(Decoder::default()))
    }

    /// Project each sample onto the named fields.
    ///
    /// Each spec may list alternatives, as in `"png;jpg;jpeg"`.
    fn to_tuple<S: AsRef<str>>(self, specs: impl IntoIterator<Item = S>) -> BoxStream<Result<Vec<Value>>> {
        let specs: Vec<String> = specs.into_iter().map(|s| s.as_ref().to_string()).collect();
        Box::pin(self.map(move |item| {
            let sample = item?;
            specs.iter().map(|spec| sample.require_first_spec(spec).cloned()).collect::<Result<Vec<Value>>>()
        }))
    }

    /// Group samples into lists of `size`, without collating them.
    fn listed(self, size: usize, partial: bool) -> BoxStream<Result<Vec<Sample>>> {
        grouped(self, size, partial)
    }

    /// Group and collate samples into batches of `size`.
    fn batched(self, size: usize, partial: bool) -> BoxStream<Result<Sample>> {
        Box::pin(grouped(self, size, partial).map(|group| collate_samples(group?)))
    }

    /// Keep each sample with probability `p`.
    fn rsample(self, p: f64, seed: Option<u64>) -> BoxStream<Result<Sample>> {
        let probability = p.clamp(0.0, 1.0);
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(entropy_seed));
        Box::pin(self.filter(move |item| {
            let keep = match item {
                Ok(_) => rng.random::<f64>() < probability,
                Err(_) => true,
            };
            async move { keep }
        }))
    }

    /// Fail with [`Error::Empty`] if the stream produced nothing.
    fn non_empty(self, message: impl Into<String>) -> BoxStream<Result<Sample>> {
        let message = message.into();

        /// What the unfold carries: the source, and whether anything arrived.
        struct State<S> {
            source: Pin<Box<S>>,
            seen: usize,
            message: String,
        }

        let state = State { source: Box::pin(self), seen: 0, message };

        Box::pin(stream::unfold(Some(state), |carried| async move {
            let mut state = carried?;
            match state.source.next().await {
                Some(item) => {
                    state.seen += 1;
                    Some((item, Some(state)))
                }
                None if state.seen == 0 => Some((Err(Error::Empty(state.message.clone())), None)),
                None => None,
            }
        }))
    }
}

impl<S: Stream<Item = Result<Sample>> + Sized + Send + 'static> AsyncSampleStreamExt for S {}

/// A boxed stream, which is what the adapters above return.
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Collect a stream into fixed-size groups.
fn grouped<S, T>(source: S, size: usize, partial: bool) -> BoxStream<Result<Vec<T>>>
where
    S: Stream<Item = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    let size = size.max(1);

    /// What the unfold carries between groups.
    struct State<S> {
        source: Pin<Box<S>>,
        /// Set once the source has ended, so it is never polled again.
        drained: bool,
    }

    let state = State { source: Box::pin(source), drained: false };

    Box::pin(stream::unfold(Some(state), move |carried| async move {
        let mut state = carried?;
        if state.drained {
            return None;
        }
        let mut group = Vec::with_capacity(size);
        while group.len() < size {
            match state.source.next().await {
                Some(Ok(item)) => group.push(item),
                Some(Err(e)) => return Some((Err(e), Some(state))),
                None => {
                    state.drained = true;
                    break;
                }
            }
        }
        match group.len() {
            0 => None,
            n if n < size && !partial => None,
            _ => Some((Ok(group), Some(state))),
        }
    }))
}

/// Adapters for streams of tuples, as produced by
/// [`to_tuple`](AsyncSampleStreamExt::to_tuple).
pub trait AsyncTupleStreamExt: Stream<Item = Result<Vec<Value>>> + Sized + Send + 'static {
    /// Group tuples into lists of `size`, without collating them.
    fn listed(self, size: usize, partial: bool) -> BoxStream<Result<Vec<Vec<Value>>>> {
        grouped(self, size, partial)
    }

    /// Group and collate tuples into batches of `size`.
    fn batched(self, size: usize, partial: bool) -> BoxStream<Result<Vec<Value>>> {
        Box::pin(grouped(self, size, partial).map(|rows| collate_tuples(rows?)))
    }

    /// Apply one function per tuple position; `None` leaves that position alone.
    fn map_tuple(self, fs: Vec<Option<crate::filters::ValueFn>>) -> BoxStream<Result<Vec<Value>>> {
        Box::pin(self.map(move |item| {
            let mut row = item?;
            for (i, f) in fs.iter().enumerate().take(row.len()) {
                if let Some(f) = f {
                    row[i] = f(row[i].clone())?;
                }
            }
            Ok(row)
        }))
    }
}

impl<S: Stream<Item = Result<Vec<Value>>> + Sized + Send + 'static> AsyncTupleStreamExt for S {}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_executor::block_on;
    use futures_util::TryStreamExt;

    fn samples(n: i64) -> impl Stream<Item = Result<Sample>> + Send + 'static {
        stream::iter((0..n).map(|i| {
            let mut s = Sample::with_key(format!("k{i}"));
            s.insert("cls", Value::Int(i));
            s.insert("txt", Value::Text(format!("t{i}")));
            Ok(s)
        }))
    }

    fn keys(stream: BoxStream<Result<Sample>>) -> Vec<String> {
        block_on(stream.try_collect::<Vec<_>>())
            .expect("no failures")
            .into_iter()
            .map(|s| s.key().expect("a key").to_string())
            .collect()
    }

    #[test]
    fn shuffle_keeps_every_sample() {
        let shuffled = keys(samples(100).shuffled(10, Some(7)));
        assert_eq!(shuffled.len(), 100);

        let mut sorted = shuffled.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 100, "no duplicates or losses");
        assert_ne!(shuffled, keys(Box::pin(samples(100))), "the order should actually change");
    }

    #[test]
    fn shuffle_matches_the_blocking_adapter_for_a_given_seed() {
        use crate::filters::SampleIteratorExt;

        let blocking: Vec<String> = (0..200)
            .map(|i| Ok(Sample::with_key(format!("k{i}"))))
            .shuffled(32, Some(99))
            .map(|s| s.expect("a sample").key().expect("a key").to_string())
            .collect();

        let asynchronous =
            keys(stream::iter((0..200).map(|i| Ok(Sample::with_key(format!("k{i}"))))).shuffled(32, Some(99)));

        assert_eq!(asynchronous, blocking, "the same seed must give the same order either way");
    }

    #[test]
    fn shuffle_handles_streams_smaller_than_the_buffer() {
        assert_eq!(keys(samples(3).shuffled(1000, Some(1))).len(), 3);
        assert_eq!(keys(samples(0).shuffled(10, Some(1))).len(), 0);
    }

    #[test]
    fn select_filters() {
        let kept = keys(samples(10).select(|s| s.get("cls").and_then(Value::as_i64).unwrap_or(0) < 3));
        assert_eq!(kept, ["k0", "k1", "k2"]);
    }

    #[test]
    fn map_drops_none_and_preserves_the_key() {
        let mapped: Vec<Sample> = block_on(
            samples(4)
                .map_sample(|s| {
                    let cls = s.get("cls").and_then(Value::as_i64).unwrap_or(0);
                    if cls % 2 == 1 {
                        return Ok(None);
                    }
                    let mut out = Sample::new();
                    out.insert("doubled", Value::Int(cls * 2));
                    Ok(Some(out))
                })
                .try_collect(),
        )
        .expect("no failures");

        assert_eq!(mapped.len(), 2);
        assert_eq!(mapped[1].key(), Some("k2"), "the key carries over");
        assert_eq!(mapped[1].get("doubled").and_then(Value::as_i64), Some(4));
    }

    #[test]
    fn map_routes_failures_to_the_handler() {
        let failing = |_: Sample| -> Result<Option<Sample>> { Err(Error::value("nope")) };

        let reraised = block_on(samples(3).map_sample(failing).collect::<Vec<_>>());
        assert!(reraised.iter().all(Result::is_err));

        let skipped = block_on(
            samples(3).map_sample_with(failing, webdataset_core::handlers::ignore_and_continue()).collect::<Vec<_>>(),
        );
        assert!(skipped.is_empty());

        let stopped = block_on(
            samples(3).map_sample_with(failing, webdataset_core::handlers::ignore_and_stop()).collect::<Vec<_>>(),
        );
        assert!(stopped.is_empty());
    }

    #[test]
    fn to_tuple_projects_fields() {
        let rows: Vec<Vec<Value>> = block_on(samples(2).to_tuple(["cls", "txt"]).try_collect()).expect("no failures");
        assert_eq!(rows[0], vec![Value::Int(0), Value::Text("t0".into())]);

        let missing = block_on(samples(2).to_tuple(["nope"]).collect::<Vec<_>>());
        assert!(missing[0].is_err());

        let alternative: Vec<Vec<Value>> =
            block_on(samples(1).to_tuple(["png;cls"]).try_collect()).expect("no failures");
        assert_eq!(alternative[0], vec![Value::Int(0)]);
    }

    #[test]
    fn listed_and_batched_group_samples() {
        let lists: Vec<Vec<Sample>> = block_on(samples(5).listed(2, true).try_collect()).expect("no failures");
        assert_eq!(lists.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2, 1]);

        let dropped: Vec<Vec<Sample>> = block_on(samples(5).listed(2, false).try_collect()).expect("no failures");
        assert_eq!(dropped.len(), 2, "the partial batch is dropped");

        let batches: Vec<Sample> = block_on(samples(4).batched(2, true).try_collect()).expect("no failures");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].get("cls").and_then(Value::as_tensor).expect("cls").shape(), &[2]);
    }

    #[test]
    fn rsample_keeps_roughly_the_requested_fraction() {
        assert_eq!(keys(samples(100).rsample(1.0, Some(1))).len(), 100);
        assert_eq!(keys(samples(100).rsample(0.0, Some(1))).len(), 0);
        let half = keys(samples(1000).rsample(0.5, Some(42))).len();
        assert!((400..600).contains(&half), "kept {half} of 1000");
    }

    #[test]
    fn non_empty_reports_an_empty_stream() {
        assert_eq!(keys(samples(3).non_empty("no data")).len(), 3);
        let outcome = block_on(samples(0).non_empty("no data").collect::<Vec<_>>());
        assert!(matches!(outcome.as_slice(), [Err(Error::Empty(_))]));
    }

    #[test]
    fn tuple_batching_collates_columns() {
        let batches: Vec<Vec<Value>> =
            block_on(samples(4).to_tuple(["cls"]).batched(2, true).try_collect()).expect("no failures");
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0][0].as_tensor().expect("a tensor").shape(), &[2]);
    }

    #[test]
    fn map_tuple_applies_per_position() {
        let double: crate::filters::ValueFn = Arc::new(|v: Value| Ok(Value::Int(v.as_i64().unwrap_or(0) * 2)));
        let rows: Vec<Vec<Value>> =
            block_on(samples(2).to_tuple(["cls", "txt"]).map_tuple(vec![Some(double), None]).try_collect())
                .expect("no failures");

        assert_eq!(rows[1][0], Value::Int(2));
        assert_eq!(rows[1][1], Value::Text("t1".into()));
    }

    #[test]
    fn errors_from_upstream_pass_through_untouched() {
        let source =
            stream::iter(vec![Ok(Sample::with_key("a")), Err(Error::value("upstream")), Ok(Sample::with_key("b"))]);
        let outcome = block_on(source.select(|_| true).collect::<Vec<_>>());
        assert_eq!(outcome.len(), 3);
        assert!(outcome[1].is_err());
    }
}
