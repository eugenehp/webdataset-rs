//! Streaming tar support for the WebDataset format.
//!
//! A WebDataset shard is an ordinary tar archive whose members are named
//! `<key>.<extension>`; the members sharing a key make up one sample. This
//! crate reads shards into [`Sample`](webdataset_core::Sample)s and writes them
//! back out, in both cases as a stream — the archive is never seeked, so the
//! same code works over HTTP, a pipe, or local disk.
//!
//! - [`TarFiles`] and [`group_by_keys`] read one shard.
//! - [`expand`] chains several shards together and marks the boundaries.
//! - [`TarWriter`] and [`ShardWriter`] write shards out.
//!
//! Compression is detected from the shard's leading bytes rather than its name,
//! so `.tar`, `.tar.gz`, and `.tgz` all just work. Enable the `zstd`, `bzip2`,
//! or `xz` features for the other containers.

#![doc(html_root_url = "https://docs.rs/webdataset-shard/0.0.1")]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// The items the `std` prelude provides, named explicitly so that the modules
/// below read the same whether or not `std` is in play.
mod prelude {
    #[allow(unused_imports)]
    pub(crate) use std::borrow::ToOwned;
    #[allow(unused_imports)]
    pub(crate) use std::boxed::Box;
    #[allow(unused_imports)]
    pub(crate) use std::format;
    #[allow(unused_imports)]
    pub(crate) use std::string::{String, ToString};
    #[allow(unused_imports)]
    pub(crate) use std::vec;
    #[allow(unused_imports)]
    pub(crate) use std::vec::Vec;
}

#[cfg(feature = "async")]
#[cfg_attr(docsrs, doc(cfg(feature = "async")))]
pub mod asynch;
pub mod compress;
pub mod reader;
pub mod sync;
pub mod writer;

pub use compress::Compression;
pub use reader::{
    Expand, FileEvent, GroupByKeys, Selection, ShardSource, TarFile, TarFiles, expand, group_by_keys, group_events,
    open_shard,
};
pub use sync::{RawEntry, TarEntries};
pub use writer::{Encoder, RawEncoder, ShardWriter, TarWriter};
