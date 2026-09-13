//! Turning shard URLs into samples.
//!
//! [`ShardsToSamples`] is the stage that does the real I/O: it opens each shard
//! URL, streams the archive, and groups the files into samples. Whether shards
//! are streamed straight through or cached on local disk first is decided by
//! the [`Opener`] it is given.
//!
//! ```no_run
//! use webdataset::pipeline::DataPipeline;
//! use webdataset::shardlists::SimpleShardList;
//! use webdataset::sources::ShardsToSamples;
//!
//! let pipeline = DataPipeline::new()
//!     .with(SimpleShardList::new(["https://host/shard-{000..009}.tar"])?)
//!     .with(ShardsToSamples::cached("./_cache"));
//! # let _ = pipeline;
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::sync::Arc;

use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{HandlerRef, reraise_exception};
use webdataset_core::sample::Sample;
use webdataset_io::cache::FileCache;
use webdataset_io::gopen::gopen;
use webdataset_io::url;
use webdataset_shard::reader::{Selection, ShardSource, expand, group_events};

use crate::pipeline::{SampleStream, Stage};

/// Opens a shard URL for reading.
pub trait Opener: Send + Sync + std::fmt::Debug {
    /// Open `url` and describe where its bytes come from.
    fn open(&self, url: &str) -> Result<ShardSource>;
}

/// Reads shards straight from their URL, without keeping a copy.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamingOpener;

impl Opener for StreamingOpener {
    fn open(&self, url: &str) -> Result<ShardSource> {
        let stream = gopen(url)?;
        let local_path = url::is_local(url).then(|| url::to_local_path(url));
        Ok(ShardSource { url: url.to_string(), local_path, stream: Box::new(FetchReader(stream)) })
    }
}

/// Downloads shards to a local cache and reads them from there.
#[derive(Debug)]
pub struct CachingOpener {
    cache: Arc<FileCache>,
}

impl CachingOpener {
    /// Cache shards under `directory`.
    pub fn new(directory: impl Into<std::path::PathBuf>) -> CachingOpener {
        CachingOpener { cache: Arc::new(FileCache::new(directory)) }
    }

    /// Use an already configured cache.
    pub fn with_cache(cache: Arc<FileCache>) -> CachingOpener {
        CachingOpener { cache }
    }

    /// The cache being used.
    pub fn cache(&self) -> &FileCache {
        &self.cache
    }
}

impl Opener for CachingOpener {
    fn open(&self, url: &str) -> Result<ShardSource> {
        let (stream, path) = self.cache.open(url)?;
        Ok(ShardSource {
            url: url.to_string(),
            local_path: Some(path.to_string_lossy().into_owned()),
            stream: Box::new(FetchReader(stream)),
        })
    }
}

/// Serves shards from bytes already in memory.
///
/// Nothing in the pipeline needs a filesystem or a subprocess once the shard
/// bytes are in hand, so this is the opener to use on WebAssembly: fetch a
/// shard however the host allows, hand the bytes over, and the rest of the
/// pipeline is unchanged.
///
/// ```
/// use std::sync::Arc;
/// use webdataset::pipeline::DataPipeline;
/// use webdataset::shardlists::SimpleShardList;
/// use webdataset::sources::{MemoryOpener, ShardsToSamples};
///
/// # let shard_bytes: Vec<u8> = Vec::new();
/// let mut opener = MemoryOpener::new();
/// opener.insert("mem://shard-000.tar", shard_bytes);
///
/// let pipeline = DataPipeline::new()
///     .with(SimpleShardList::verbatim(["mem://shard-000.tar"]))
///     .with(ShardsToSamples::new(Arc::new(opener)));
/// # let _ = pipeline;
/// ```
#[derive(Debug, Default)]
pub struct MemoryOpener {
    shards: std::collections::HashMap<String, std::sync::Arc<[u8]>>,
}

impl MemoryOpener {
    /// An opener with no shards.
    pub fn new() -> MemoryOpener {
        MemoryOpener::default()
    }

    /// Serve `bytes` for `url`.
    pub fn insert(&mut self, url: impl Into<String>, bytes: impl Into<std::sync::Arc<[u8]>>) {
        self.shards.insert(url.into(), bytes.into());
    }

    /// Serve `bytes` for `url`, consuming and returning the opener.
    pub fn with(mut self, url: impl Into<String>, bytes: impl Into<std::sync::Arc<[u8]>>) -> MemoryOpener {
        self.insert(url, bytes);
        self
    }

    /// The URLs this opener can serve.
    pub fn urls(&self) -> impl Iterator<Item = &str> {
        self.shards.keys().map(String::as_str)
    }
}

impl Opener for MemoryOpener {
    fn open(&self, url: &str) -> Result<ShardSource> {
        let bytes =
            self.shards.get(url).ok_or_else(|| Error::value(format!("{url} is not held by this in-memory opener")))?;
        Ok(ShardSource {
            url: url.to_string(),
            local_path: None,
            stream: Box::new(std::io::Cursor::new(bytes.to_vec())),
        })
    }
}

/// Adapts a [`Fetch`](webdataset_io::gopen::Fetch) into a plain reader.
struct FetchReader(Box<dyn webdataset_io::gopen::Fetch>);

impl std::io::Read for FetchReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

/// Reads shards and yields the samples inside them.
#[derive(Debug)]
pub struct ShardsToSamples {
    opener: Arc<dyn Opener>,
    selection: Selection,
    handler: HandlerRef,
}

impl ShardsToSamples {
    /// Read shards with `opener`.
    pub fn new(opener: Arc<dyn Opener>) -> ShardsToSamples {
        ShardsToSamples { opener, selection: Selection::new(), handler: reraise_exception() }
    }

    /// Read shards straight from their URLs.
    pub fn streaming() -> ShardsToSamples {
        ShardsToSamples::new(Arc::new(StreamingOpener))
    }

    /// Cache shards under `directory` and read them from there.
    pub fn cached(directory: impl Into<std::path::PathBuf>) -> ShardsToSamples {
        ShardsToSamples::new(Arc::new(CachingOpener::new(directory)))
    }

    /// Choose or rename the files read out of each shard.
    pub fn with_selection(mut self, selection: Selection) -> ShardsToSamples {
        self.selection = selection;
        self
    }

    /// Decide what happens when a shard cannot be read.
    pub fn with_handler(mut self, handler: HandlerRef) -> ShardsToSamples {
        self.handler = handler;
        self
    }
}

impl Stage for ShardsToSamples {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let opener = self.opener.clone();
        let sources = input.map(move |item| {
            let sample: Sample = item?;
            let url =
                sample.url().ok_or_else(|| Error::value("shard list produced a sample without a __url__"))?.to_string();
            opener.open(&url).map_err(|e| e.context(url))
        });
        let files = expand(sources, self.selection.clone(), self.handler.clone());
        Box::new(group_events(files, self.handler.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::DataPipeline;
    use crate::shardlists::SimpleShardList;

    fn testdata(name: &str) -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn reads_samples_out_of_a_shard() {
        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim([testdata("sample.tgz")]))
            .with(ShardsToSamples::streaming());

        let samples: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
        assert!(!samples.is_empty());
        assert!(samples[0].contains_key("png"));
        assert!(samples[0].url().unwrap().ends_with("sample.tgz"));
        assert!(samples[0].local_path().is_some(), "local shards report their path");
    }

    #[test]
    fn reads_several_shards_in_order() {
        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim([testdata("sample.tgz"), testdata("mpdata.tar")]))
            .with(ShardsToSamples::streaming());

        let urls: Vec<String> = pipeline.iter().map(|s| s.unwrap().url().unwrap().to_string()).collect();
        assert!(urls.first().unwrap().ends_with("sample.tgz"));
        assert!(urls.last().unwrap().ends_with("mpdata.tar"));
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn caches_shards_when_asked() {
        let cache = tempfile::tempdir().unwrap();
        let url = format!("pipe:cat {}", testdata("sample.tgz"));

        let pipeline =
            DataPipeline::new().with(SimpleShardList::verbatim([url])).with(ShardsToSamples::cached(cache.path()));

        assert!(pipeline.iter().count() > 0);
        assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 1, "the shard should have been cached");
    }

    #[test]
    fn reports_unreadable_shards_through_the_handler() {
        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim(["/no/such/shard.tar"]))
            .with(ShardsToSamples::streaming());
        assert!(pipeline.iter().next().unwrap().is_err());

        let skipping = DataPipeline::new()
            .with(SimpleShardList::verbatim(["/no/such/shard.tar".to_string(), testdata("sample.tgz")]))
            .with(ShardsToSamples::streaming().with_handler(webdataset_core::handlers::ignore_and_continue()));
        assert!(skipping.iter().count() > 0, "the good shard should still be read");
    }

    #[test]
    fn serves_shards_from_memory() {
        let bytes = std::fs::read(testdata("sample.tgz")).unwrap();
        let opener = MemoryOpener::new().with("mem://a.tar", bytes);

        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim(["mem://a.tar"]))
            .with(ShardsToSamples::new(Arc::new(opener)));

        let samples: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), 90);
        assert_eq!(samples[0].url(), Some("mem://a.tar"));
        assert!(samples[0].local_path().is_none(), "an in-memory shard has no path");
    }

    #[test]
    fn reports_a_shard_the_memory_opener_does_not_hold() {
        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim(["mem://missing.tar"]))
            .with(ShardsToSamples::new(Arc::new(MemoryOpener::new())));
        assert!(pipeline.iter().next().unwrap().is_err());
    }

    #[test]
    fn applies_a_file_selection() {
        let selection = Selection::new().select(|name| name.ends_with(".cls"));
        let pipeline = DataPipeline::new()
            .with(SimpleShardList::verbatim([testdata("sample.tgz")]))
            .with(ShardsToSamples::streaming().with_selection(selection));

        let samples: Vec<Sample> = pipeline.iter().map(|s| s.unwrap()).collect();
        assert!(!samples.is_empty());
        assert!(samples.iter().all(|s| s.field_names() == ["cls"]));
    }
}
