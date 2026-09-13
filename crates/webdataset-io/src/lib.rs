//! I/O for the WebDataset format: opening shard URLs and caching them locally.
//!
//! The entry point is [`gopen()`], which turns any shard URL into a byte stream:
//!
//! | scheme | handled by |
//! |---|---|
//! | *(none)*, `file:` | the local filesystem |
//! | `pipe:` | the system shell |
//! | `http:`, `https:`, `ftp:`, `ftps:`, `sftp:`, `scp:` | `curl` |
//! | `gs:` | `gsutil` |
//! | `htgs:` | `curl`, against `storage.googleapis.com` |
//! | `ais:` | `ais` |
//! | `hf:` | `curl`, against `huggingface.co` |
//!
//! Shelling out keeps credentials and proxy configuration in the transfer
//! tools, exactly as the Python implementation does. Add your own schemes with
//! [`gopen::register_scheme`].
//!
//! Every scheme except the local filesystem needs a subprocess, so all of them
//! live behind the `subprocess` feature. Turning it off leaves the local
//! filesystem and whatever schemes you register yourself, which is the shape
//! this crate takes on WebAssembly.
//!
//! [`cache::FileCache`] layers a local shard cache on top, with
//! atomic downloads, format validation, and LRU eviction.

#![doc(html_root_url = "https://docs.rs/webdataset-io/0.0.1")]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod cache;
pub mod gopen;
#[cfg(feature = "subprocess")]
#[cfg_attr(docsrs, doc(cfg(feature = "subprocess")))]
pub mod pipe;
pub mod url;

pub use cache::{FileCache, LruCleanup, url_to_cache_name};
pub use gopen::{Fetch, SchemeHandler, Sink, gopen, gopen_write, register_scheme};
#[cfg(feature = "subprocess")]
pub use pipe::Pipe;
