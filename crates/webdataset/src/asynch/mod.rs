//! Asynchronous pipelines.
//!
//! This is the same library with the blocking calls taken out: the same archive
//! parser, the same decoders, the same shuffling and batching, driven by
//! futures and producing a [`Stream`](futures_core::Stream) rather than an
//! [`Iterator`].
//!
//! Use it when shards arrive over a network. A blocking reader occupies a
//! thread for the whole of a transfer; an async one occupies a task, so one
//! thread can keep many shards in flight — which is exactly what hides the
//! latency of fetching from object storage.
//!
//! ```
//! use futures_util::TryStreamExt;
//! use webdataset::asynch::{AsyncSampleStreamExt, AsyncWebDataset, MemoryOpener};
//!
//! # futures_executor::block_on(async {
//! # let bytes = std::fs::read("../../testdata/imagenet-000000.tgz").unwrap();
//! let opener = MemoryOpener::new().with("mem://shard-000.tar", bytes);
//!
//! let dataset = AsyncWebDataset::builder_verbatim(["mem://shard-000.tar"])
//!     .opener(std::sync::Arc::new(opener))
//!     .concurrency(4)
//!     .build()?
//!     .shuffle(1000)
//!     .decode_basic();
//!
//! let batches: Vec<_> = dataset.stream().batched(16, true).try_collect().await?;
//! assert_eq!(batches.len(), 3);
//! # Ok::<(), webdataset_core::Error>(())
//! # }).unwrap();
//! ```
//!
//! # What is and is not async
//!
//! Only the I/O is. Decoding a JPEG, stacking a batch, and shuffling a buffer
//! are computations; making them `async` would add overhead and no concurrency.
//! They are the same code the blocking pipeline runs, which is why the two
//! produce identical samples — a property the test suite checks shard by shard.
//!
//! For CPU-bound work that genuinely needs parallelism, map it onto your
//! runtime's blocking pool before it reaches the stream.
//!
//! # Transports
//!
//! Anything implementing [`futures_io::AsyncRead`] can be a shard. Two openers
//! are built in — [`MemoryOpener`] for bytes you already hold and
//! [`FileOpener`] for local paths — and [`AsyncOpener`] is a two-line trait for
//! anything else, such as an HTTP client's byte stream.

mod dataset;
mod filters;
mod pipeline;
mod sources;
pub mod stages;

pub use dataset::{AsyncWebDataset, AsyncWebDatasetBuilder};
pub use filters::{AsyncSampleStreamExt, AsyncTupleStreamExt};
pub use pipeline::{AsyncDataPipeline, AsyncStage, LiftSource, SampleStream, empty_stream, stage_fn};
pub use sources::{AsyncOpener, AsyncShardSource, AsyncShardsToSamples, BoxFuture, FileOpener, MemoryOpener};
pub use stages::{CheckEmpty, SingleNodeOnly, SplitByNode, SplitByWorker};
