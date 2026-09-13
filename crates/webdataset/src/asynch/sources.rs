//! Getting shard bytes, without blocking.

use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use futures_util::stream::{self, StreamExt};
use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{Action, HandlerRef, reraise_exception};
use webdataset_shard::Selection;
use webdataset_shard::asynch::{Source, shard_samples};

use super::pipeline::{AsyncStage, SampleStream};

/// A future that can be named, which is what a `dyn` trait method needs.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// An opened shard: where it came from and the bytes behind it.
pub struct AsyncShardSource {
    /// The URL the shard was named by.
    pub url: String,
    /// Where the shard lives on disk, if it does.
    pub local_path: Option<String>,
    /// The shard's bytes.
    pub stream: Box<dyn Source>,
}

impl std::fmt::Debug for AsyncShardSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncShardSource").field("url", &self.url).field("local_path", &self.local_path).finish()
    }
}

/// Opens a shard URL without blocking.
///
/// Implement this for whatever transport you have. An HTTP client whose
/// response body is a `Stream<Item = io::Result<Bytes>>` becomes a shard source
/// with `futures_util::TryStreamExt::into_async_read`.
pub trait AsyncOpener: Send + Sync + std::fmt::Debug {
    /// Open `url`.
    fn open<'a>(&'a self, url: &'a str) -> BoxFuture<'a, Result<AsyncShardSource>>;
}

/// Serves shards from bytes already in memory.
///
/// The opener to reach for on WebAssembly, and the one the tests use: fetch a
/// shard however the host allows, hand the bytes over, and everything above is
/// unchanged.
#[derive(Debug, Default)]
pub struct MemoryOpener {
    shards: HashMap<String, Arc<[u8]>>,
}

impl MemoryOpener {
    /// An opener with no shards.
    pub fn new() -> MemoryOpener {
        MemoryOpener::default()
    }

    /// Serve `bytes` for `url`.
    pub fn insert(&mut self, url: impl Into<String>, bytes: impl Into<Arc<[u8]>>) {
        self.shards.insert(url.into(), bytes.into());
    }

    /// Serve `bytes` for `url`, consuming and returning the opener.
    pub fn with(mut self, url: impl Into<String>, bytes: impl Into<Arc<[u8]>>) -> MemoryOpener {
        self.insert(url, bytes);
        self
    }

    /// The URLs this opener can serve.
    pub fn urls(&self) -> impl Iterator<Item = &str> {
        self.shards.keys().map(String::as_str)
    }
}

impl AsyncOpener for MemoryOpener {
    fn open<'a>(&'a self, url: &'a str) -> BoxFuture<'a, Result<AsyncShardSource>> {
        Box::pin(async move {
            let bytes = self
                .shards
                .get(url)
                .ok_or_else(|| Error::value(format!("{url} is not held by this in-memory opener")))?;
            Ok(AsyncShardSource {
                url: url.to_string(),
                local_path: None,
                stream: Box::new(futures_util::io::Cursor::new(bytes.to_vec())),
            })
        })
    }
}

/// Serves shards from the local filesystem.
///
/// The read itself is a blocking one wrapped in
/// [`AllowStdIo`](futures_util::io::AllowStdIo): the filesystem has no portable
/// async interface, and a local read does not benefit from one. On a runtime
/// with a blocking pool, or with `tokio::fs`, supply your own opener instead —
/// this one will stall the executor on a slow disk.
#[derive(Debug, Default, Clone)]
pub struct FileOpener {
    root: Option<PathBuf>,
}

impl FileOpener {
    /// Open paths as given.
    pub fn new() -> FileOpener {
        FileOpener::default()
    }

    /// Resolve relative paths against `root`.
    pub fn rooted(root: impl Into<PathBuf>) -> FileOpener {
        FileOpener { root: Some(root.into()) }
    }
}

impl AsyncOpener for FileOpener {
    fn open<'a>(&'a self, url: &'a str) -> BoxFuture<'a, Result<AsyncShardSource>> {
        Box::pin(async move {
            let path = webdataset_io::url::to_local_path(url);
            let path = match &self.root {
                Some(root) => root.join(path),
                None => PathBuf::from(path),
            };
            let file =
                std::fs::File::open(&path).map_err(|e| Error::Io(e).context(format!("opening {}", path.display())))?;
            Ok(AsyncShardSource {
                url: url.to_string(),
                local_path: Some(path.to_string_lossy().into_owned()),
                stream: Box::new(futures_util::io::AllowStdIo::new(std::io::BufReader::new(file))),
            })
        })
    }
}

/// Reads shards and yields the samples inside them.
///
/// With `concurrency` above one, that many shards are fetched and parsed at the
/// same time and their samples interleave, which is how the latency of a remote
/// fetch is hidden. The interleaving order is unspecified, so a concurrent
/// stream is not reproducible sample-for-sample; leave it at one when that
/// matters.
#[derive(Debug)]
pub struct AsyncShardsToSamples {
    opener: Arc<dyn AsyncOpener>,
    selection: Selection,
    handler: HandlerRef,
    concurrency: usize,
}

impl AsyncShardsToSamples {
    /// Read shards with `opener`, one at a time.
    pub fn new(opener: Arc<dyn AsyncOpener>) -> AsyncShardsToSamples {
        AsyncShardsToSamples { opener, selection: Selection::new(), handler: reraise_exception(), concurrency: 1 }
    }

    /// Read shards from bytes held in memory.
    pub fn memory(opener: MemoryOpener) -> AsyncShardsToSamples {
        AsyncShardsToSamples::new(Arc::new(opener))
    }

    /// Read shards from the local filesystem.
    pub fn files() -> AsyncShardsToSamples {
        AsyncShardsToSamples::new(Arc::new(FileOpener::new()))
    }

    /// Keep this many shards in flight at once.
    pub fn with_concurrency(mut self, concurrency: usize) -> AsyncShardsToSamples {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Choose or rename the files read out of each shard.
    pub fn with_selection(mut self, selection: Selection) -> AsyncShardsToSamples {
        self.selection = selection;
        self
    }

    /// Decide what happens when a shard cannot be read.
    pub fn with_handler(mut self, handler: HandlerRef) -> AsyncShardsToSamples {
        self.handler = handler;
        self
    }
}

impl AsyncStage for AsyncShardsToSamples {
    fn apply(&self, input: SampleStream) -> SampleStream {
        let opener = self.opener.clone();
        let selection = self.selection.clone();
        let handler = self.handler.clone();
        let concurrency = self.concurrency;

        // One stream of samples per shard, then flattened. The open happens
        // *inside* each inner stream rather than while building it, so that
        // `flatten_unordered` drives several opens at once — which is what
        // overlaps the fetches. Awaiting the open out here instead would
        // serialise them however high the concurrency was set.
        let per_shard = input.map(move |item| -> SampleStream {
            let opener = opener.clone();
            let selection = selection.clone();
            let handler = handler.clone();

            let opening = stream::once(async move {
                match item {
                    Ok(sample) => match sample.url() {
                        Some(url) => opener.open(url).await.map_err(|e| e.context(url.to_string())),
                        None => Err(Error::value("shard list produced a sample without a __url__")),
                    },
                    Err(e) => Err(e),
                }
            });

            Box::pin(opening.flat_map(move |opened| -> SampleStream {
                match opened {
                    Ok(source) => shard_samples(
                        source.stream,
                        Some(source.url),
                        source.local_path,
                        selection.clone(),
                        handler.clone(),
                    ),
                    Err(e) => match handler.handle(&e) {
                        // An empty shard stream is how "skip this one" and
                        // "stop here" are both expressed.
                        Action::Continue | Action::Stop => Box::pin(stream::empty()),
                        Action::Reraise => Box::pin(stream::once(async move { Err(e) })),
                    },
                }
            }))
        });

        Box::pin(per_shard.flatten_unordered(concurrency))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asynch::AsyncDataPipeline;
    use crate::shardlists::SimpleShardList;
    use futures_executor::block_on;
    use futures_util::TryStreamExt;
    use webdataset_core::Sample;

    fn shard(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
        std::fs::read(path).expect("reading the shard")
    }

    fn testdata(name: &str) -> String {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata")
            .join(name)
            .to_string_lossy()
            .into_owned()
    }

    fn collect(pipeline: &AsyncDataPipeline) -> Vec<Sample> {
        block_on(pipeline.stream().try_collect::<Vec<_>>()).expect("no failures")
    }

    #[test]
    fn reads_shards_from_memory() {
        let opener = MemoryOpener::new().with("mem://a.tar", shard("sample.tgz"));
        let pipeline = AsyncDataPipeline::new()
            .with_source(SimpleShardList::verbatim(["mem://a.tar"]))
            .with(AsyncShardsToSamples::memory(opener));

        let samples = collect(&pipeline);
        assert_eq!(samples.len(), 90);
        assert_eq!(samples[0].url(), Some("mem://a.tar"));
        assert!(samples[0].local_path().is_none());
    }

    #[test]
    fn reads_shards_from_the_filesystem() {
        let pipeline = AsyncDataPipeline::new()
            .with_source(SimpleShardList::verbatim([testdata("imagenet-000000.tgz")]))
            .with(AsyncShardsToSamples::files());

        let samples = collect(&pipeline);
        assert_eq!(samples.len(), 47);
        assert!(samples[0].local_path().is_some());
    }

    #[test]
    fn concurrency_does_not_change_the_sample_set() {
        let opener = || {
            MemoryOpener::new()
                .with("mem://a.tar", shard("sample.tgz"))
                .with("mem://b.tar", shard("imagenet-000000.tgz"))
                .with("mem://c.tar", shard("mpdata.tar"))
        };
        let build = |concurrency: usize| {
            AsyncDataPipeline::new()
                .with_source(SimpleShardList::verbatim(["mem://a.tar", "mem://b.tar", "mem://c.tar"]))
                .with(AsyncShardsToSamples::memory(opener()).with_concurrency(concurrency))
        };

        let mut sequential: Vec<String> =
            collect(&build(1)).iter().map(|s| s.key().expect("a key").to_string()).collect();
        let mut concurrent: Vec<String> =
            collect(&build(3)).iter().map(|s| s.key().expect("a key").to_string()).collect();

        assert_eq!(sequential.len(), 90 + 47 + 100);
        sequential.sort();
        concurrent.sort();
        assert_eq!(sequential, concurrent, "concurrency may reorder, never lose or duplicate");
    }

    #[test]
    fn reads_shards_in_order_when_sequential() {
        let opener =
            MemoryOpener::new().with("mem://a.tar", shard("sample.tgz")).with("mem://b.tar", shard("mpdata.tar"));
        let pipeline = AsyncDataPipeline::new()
            .with_source(SimpleShardList::verbatim(["mem://a.tar", "mem://b.tar"]))
            .with(AsyncShardsToSamples::memory(opener));

        let urls: Vec<String> = collect(&pipeline).iter().map(|s| s.url().expect("a url").to_string()).collect();
        assert_eq!(urls.first().map(String::as_str), Some("mem://a.tar"));
        assert_eq!(urls.last().map(String::as_str), Some("mem://b.tar"));
    }

    #[test]
    fn reports_a_shard_the_opener_does_not_hold() {
        let pipeline = AsyncDataPipeline::new()
            .with_source(SimpleShardList::verbatim(["mem://missing.tar"]))
            .with(AsyncShardsToSamples::memory(MemoryOpener::new()));
        let outcome: Vec<_> = block_on(pipeline.stream().collect::<Vec<_>>());
        assert_eq!(outcome.len(), 1);
        assert!(outcome[0].is_err());
    }

    #[test]
    fn skips_an_unreadable_shard_when_asked() {
        let opener = MemoryOpener::new().with("mem://good.tar", shard("sample.tgz"));
        let pipeline = AsyncDataPipeline::new()
            .with_source(SimpleShardList::verbatim(["mem://missing.tar", "mem://good.tar"]))
            .with(AsyncShardsToSamples::memory(opener).with_handler(webdataset_core::handlers::ignore_and_continue()));
        assert_eq!(collect(&pipeline).len(), 90);
    }

    #[test]
    fn applies_a_file_selection() {
        let opener = MemoryOpener::new().with("mem://a.tar", shard("sample.tgz"));
        let pipeline = AsyncDataPipeline::new().with_source(SimpleShardList::verbatim(["mem://a.tar"])).with(
            AsyncShardsToSamples::memory(opener).with_selection(Selection::new().select(|name| name.ends_with(".cls"))),
        );
        let samples = collect(&pipeline);
        assert_eq!(samples.len(), 90);
        assert!(samples.iter().all(|s| s.field_names() == ["cls"]));
    }
}
