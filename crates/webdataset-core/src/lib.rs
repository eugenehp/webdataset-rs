//! Core data model for the [WebDataset](https://github.com/webdataset/webdataset) format.
//!
//! WebDataset stores a dataset as a set of POSIX tar archives ("shards"). The
//! files inside a shard that share a basename make up one training sample:
//!
//! ```text
//! shard-000000.tar
//!   image0001.jpg   image0001.cls   image0001.json
//!   image0002.jpg   image0002.cls   image0002.json
//! ```
//!
//! This crate defines the pieces every other crate in the workspace shares:
//!
//! - [`Sample`] and [`Value`] — the in-memory shape of a training example.
//! - [`Tensor`] and [`DType`] — dense numeric arrays, plus the [`npy`] codec.
//! - [`Error`] and [`handlers`] — how failures are reported and absorbed.
//! - [`braceexpand()`] and [`utils`] — shard-list expansion and worker identity.
//!
//! Most users should depend on the `webdataset` crate instead, which
//! re-exports everything here.
//!
//! # `no_std`
//!
//! Turning off the `std` feature builds against `core` and `alloc` only, which
//! covers the whole data model: samples, values, tensors, the `.npy` codec,
//! brace expansion, and the error and handler types. What `std` adds is the
//! parts that need an operating system — `Error::Io`, environment variable
//! substitution, reading worker identity from the environment, and the
//! stream-based `npy` helpers.
//!
//! ```toml
//! webdataset-core = { version = "0.1", default-features = false, features = ["json"] }
//! ```
//!
//! # Threads
//!
//! The `threads` feature — on by default — makes per-worker state thread-safe.
//! With the standard library that is a thread-local and needs no setup. Without
//! it, there is no portable way to ask which thread is running, so the host
//! supplies one with
//! [`workers::set_thread_id_hook`]; see the [`workers`]
//! module. A single-threaded `no_std` program needs neither.

#![doc(html_root_url = "https://docs.rs/webdataset-core/0.0.1")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// The items the `std` prelude would provide, sourced from `alloc` so that the
/// same code compiles with and without `std`.
mod prelude {
    #[allow(unused_imports)]
    pub(crate) use alloc::borrow::ToOwned;
    #[allow(unused_imports)]
    pub(crate) use alloc::boxed::Box;
    #[allow(unused_imports)]
    pub(crate) use alloc::format;
    #[allow(unused_imports)]
    pub(crate) use alloc::string::{String, ToString};
    #[allow(unused_imports)]
    pub(crate) use alloc::vec;
    #[allow(unused_imports)]
    pub(crate) use alloc::vec::Vec;
}

pub mod braceexpand;
pub mod error;
pub mod fields;
pub mod handlers;
pub mod npy;
pub mod sample;
pub mod tensor;
pub mod utils;
pub mod value;
pub mod workers;

pub use braceexpand::{braceexpand, braceexpand_all};
pub use error::{Error, Result, ResultExt};
pub use fields::{FieldHasher, Fields};
pub use handlers::{Action, Dispatch, Handler, HandlerRef};
pub use sample::Sample;
pub use tensor::{DType, Tensor};
pub use utils::{WorkerInfo, base_plus_ext, expand_urls, worker_info};
pub use value::{CustomValue, Value};
pub use workers::{set_thread_id_hook, set_worker, with_worker};
