//! High-performance sequential dataset loading, in the
//! [WebDataset](https://github.com/webdataset/webdataset) format.
//!
//! A WebDataset is a set of tar archives ("shards"). Inside a shard, the files
//! that share a basename make up one training sample:
//!
//! ```text
//! imagenet-000000.tar
//!   n03991062_24866.jpg   n03991062_24866.cls
//!   n03995372_9042.jpg    n03995372_9042.cls
//! ```
//!
//! Reading is purely sequential, which is what makes the format fast: no seeks,
//! no index, no per-file requests. A shard reads at the full bandwidth of the
//! device or the network link, and a dataset is just a list of URLs — nothing
//! needs to be registered, converted, or mounted first.
//!
//! # Reading a dataset
//!
//! ```no_run
//! use webdataset::{Decoder, WebDataset};
//! use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
//!
//! let dataset = WebDataset::builder("https://host/imagenet-{000000..000146}.tar")
//!     .shard_shuffle(100)
//!     .build()?
//!     .shuffle(1000)
//!     .decode(Decoder::default());
//!
//! for batch in dataset.iter().to_tuple(["jpg;png", "cls"]).batched(64, true) {
//!     let batch = batch?;
//!     let images = &batch[0];
//!     let labels = &batch[1];
//!     # let _ = (images, labels);
//! }
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! # Writing a dataset
//!
//! ```
//! use std::sync::Arc;
//! use webdataset::encode::DefaultEncoder;
//! use webdataset::{Sample, ShardWriter, Value};
//!
//! # let dir = tempfile::tempdir()?;
//! # let pattern = dir.path().join("train-%06d.tar");
//! # let pattern = pattern.to_str().unwrap();
//! let mut writer = ShardWriter::new(pattern)?
//!     .with_encoder(Arc::new(DefaultEncoder::new()))
//!     .with_max_count(10_000);
//!
//! for i in 0..100 {
//!     let mut sample = Sample::with_key(format!("sample{i:06}"));
//!     sample.insert("cls", Value::Int(i % 10));
//!     sample.insert("txt", Value::Text(format!("sample number {i}")));
//!     writer.write(&sample)?;
//! }
//! writer.close()?;
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! # How a pipeline is put together
//!
//! [`WebDataset`] assembles the usual stages for you, but the pieces are public
//! and a pipeline can be built by hand:
//!
//! ```text
//! SimpleShardList      a list of shard URLs
//!   -> SplitByNode     keep this rank's shards
//!   -> SplitByWorker   keep this worker's shards
//!   -> Shuffle         shuffle the shard order
//!   -> ShardsToSamples open each shard, group its files into samples
//!   -> Shuffle         shuffle samples within a buffer
//!   -> Decode          turn bytes into images, tensors, JSON
//! ```
//!
//! Everything up to this point is a [`Stage`] producing
//! [`Sample`]s, so it can be re-run each epoch. Transformations that change the
//! item type — projecting to tuples, batching — are ordinary iterator adapters
//! from [`filters`], applied to [`WebDataset::iter()`].
//!
//! # Scaling out
//!
//! [`DataLoader`] runs one copy of the pipeline per worker thread. Shards are
//! divided between workers by the [`SplitByWorker`] stage, and between
//! distributed processes by [`SplitByNode`]. When exact partitioning is
//! awkward — many nodes, few shards — use `resampled(true)` instead and let
//! each worker draw shards with replacement.
//!
//! # Errors
//!
//! Every stream yields `Result<Sample>`. What happens when a shard is corrupt
//! or a field fails to decode is decided by a
//! [`Handler`]: forward the error, drop the sample,
//! or end the stream. Streaming a petabyte means meeting some bad bytes, so
//! `warn_and_continue` is a common choice.
//!
//! # Features
//!
//! | feature | adds |
//! |---|---|
//! | `threads` *(default)* | per-shard read-ahead and the multi-worker loader |
//! | `subprocess` *(default)* | `pipe:` and the `curl`/`gsutil`/`ais` schemes |
//! | `yaml` *(default)* | multi-source dataset specifications |
//! | `image` | `.jpg`, `.png` and friends, via the `image` crate |
//! | `msgpack` | `.mp` and `.msg` |
//! | `cbor` | `.cbor` |
//! | `npz` | NumPy `.npz` archives |
//! | `zstd`, `bzip2`, `xz` | shards in those containers |
//! | `async` | read shards from any `AsyncRead`, yielding a `Stream` |
//! | `wasm-js` | host randomness on `wasm32-unknown-unknown` |
//! | `full` | every format, plus threads, subprocesses and async |
//!
//! `.npy`, `.ten`, `.json`, `.txt`, `.cls` and gzip need no features.
//!
//! # Async
//!
//! With the `async` feature, [`asynch`] mirrors everything above: the same
//! archive parser, the same decoders, the same shuffling and batching, driven
//! by futures and producing a [`Stream`](futures_core::Stream) rather than an
//! [`Iterator`]. Reach for it when shards arrive over a network — a blocking
//! reader holds a thread for the whole of a transfer, an async one holds only a
//! task, and raising
//! [`concurrency`](asynch::AsyncWebDatasetBuilder::concurrency) overlaps the
//! fetches.
//!
//! Only the I/O is async; decoding and batching are the same code the blocking
//! pipeline runs, which is why the two read identically — a property the test
//! suite checks shard by shard. See the [`asynch`] module for a worked example.
//!
//! # WebAssembly
//!
//! WebAssembly has no threads to spawn and no processes to run, so turn both
//! features off and hand the shard bytes over yourself with
//! [`MemoryOpener`]. Everything above the transport — the archive parser, the
//! decoders, shuffling, batching — is unchanged.
//!
//! ```
//! use std::sync::Arc;
//! use webdataset::pipeline::DataPipeline;
//! use webdataset::shardlists::SimpleShardList;
//! use webdataset::sources::{MemoryOpener, ShardsToSamples};
//! use webdataset::stages::Decode;
//!
//! # let bytes: Vec<u8> = std::fs::read("../../testdata/sample.tgz")?;
//! let opener = MemoryOpener::new().with("mem://shard-000.tar", bytes);
//! let dataset = DataPipeline::new()
//!     .with(SimpleShardList::verbatim(["mem://shard-000.tar"]))
//!     .with(ShardsToSamples::new(Arc::new(opener)))
//!     .with(Decode::basic());
//!
//! assert_eq!(dataset.iter().count(), 90);
//! # Ok::<(), webdataset_core::Error>(())
//! ```
//!
//! For any other transport, register a scheme with
//! [`webdataset_io::register_scheme`].
//!
//! `webdataset-core` and `webdataset-tenbin` go further and build without the
//! standard library at all; see their documentation.

#![doc(html_root_url = "https://docs.rs/webdataset/0.0.1")]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod asynch;
pub mod batch;
pub mod dataset;
pub mod decode;
pub mod encode;
pub mod filters;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod images;
pub mod loader;
pub mod mix;
pub mod pipeline;
pub mod shardlists;
pub mod sources;
pub mod stages;

pub mod prelude {
    //! The imports a typical pipeline needs, in one line.
    //!
    //! ```
    //! use webdataset::prelude::*;
    //! ```
    //!
    //! This is a convenience, not the whole API: it carries the types you name
    //! when building and running a pipeline, and the traits whose methods would
    //! otherwise be invisible. Anything more specialised — the shard list
    //! implementations, the mixers, the encoder — is still reached through its
    //! own module, so glob-importing this does not pull the crate into scope.
    //!
    //! Traits first, since those are the ones that fail confusingly when
    //! missing: without [`SampleIteratorExt`] in scope, `.shuffle(...)` on a
    //! stream of samples is simply not a method.

    pub use crate::filters::{SampleIteratorExt, TupleIteratorExt};
    pub use crate::pipeline::{DataPipeline, SampleStream, Stage};

    pub use crate::dataset::{NodeSplit, WebDataset, WebDatasetBuilder};
    pub use crate::decode::{Decoded, Decoder};
    pub use crate::loader::DataLoader;

    pub use webdataset_core::{
        Action, DType, Error, Handler, HandlerRef, Result, Sample, Tensor, Value, WorkerInfo, handlers,
    };
    pub use webdataset_shard::{Selection, ShardWriter, TarWriter};

    /// Reading an `imagespec` such as `"rgb8"` needs this in scope.
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub use crate::images::ImageSpec;

    /// The async equivalents, where the names do not clash.
    ///
    /// `asynch::SampleStream` is deliberately left out: it is a different type
    /// from the blocking [`SampleStream`] above and sharing a name in one glob
    /// import would be a trap. Reach for it as `webdataset::asynch::SampleStream`.
    #[cfg(feature = "async")]
    #[cfg_attr(docsrs, doc(cfg(feature = "async")))]
    pub use crate::asynch::{
        AsyncDataPipeline, AsyncSampleStreamExt, AsyncStage, AsyncTupleStreamExt, AsyncWebDataset,
    };
}

pub use batch::{collate_samples, collate_tuples, uncollate_sample};
pub use dataset::{NodeSplit, WebDataset, WebDatasetBuilder};
pub use decode::{DecodeHandler, Decoded, Decoder};
pub use encode::DefaultEncoder;
pub use filters::{SampleIteratorExt, TupleIteratorExt};
pub use loader::DataLoader;
pub use mix::{RandomMix, RoundRobin};
pub use pipeline::{DataPipeline, SampleStream, Stage};
pub use shardlists::{ResampledShards, SimpleShardList, SplitByNode, SplitByWorker};
pub use sources::{MemoryOpener, Opener, ShardsToSamples};

#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub use images::{ImageHandler, ImageSpec};

/// The core data model: samples, values, tensors, and errors.
pub use webdataset_core as core;
/// URL opening and shard caching.
pub use webdataset_io as io;
/// Streaming tar readers and writers.
pub use webdataset_shard as shard;
/// The `.ten` binary tensor format.
pub use webdataset_tenbin as tenbin;

pub use webdataset_core::{
    Action, DType, Error, Handler, HandlerRef, Result, Sample, Tensor, Value, WorkerInfo, braceexpand, handlers,
};
pub use webdataset_io::{FileCache, gopen, gopen_write};
pub use webdataset_shard::{Selection, ShardWriter, TarWriter};
