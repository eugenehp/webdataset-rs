//! Reading shards from any [`AsyncRead`].
//!
//! This mirrors the blocking reader in [`reader`](crate::reader) exactly — same
//! archive parser, same grouping rule, same shard-boundary handling — but takes
//! its bytes from a future rather than a blocking call, and hands back a
//! [`Stream`] instead of an [`Iterator`].
//!
//! That matters where a shard arrives over a network: a blocking reader holds a
//! thread for the whole transfer, whereas an async one holds only a task, so a
//! single thread can keep many shards in flight.
//!
//! ```
//! use futures_util::StreamExt;
//! use futures_util::io::Cursor;
//! use webdataset_shard::asynch::AsyncShardFiles;
//! use webdataset_shard::{Selection, asynch::group_events_stream};
//! use webdataset_core::handlers::reraise_exception;
//!
//! # futures_executor::block_on(async {
//! # let bytes = std::fs::read("../../testdata/sample.tgz").unwrap();
//! let files = AsyncShardFiles::new(Cursor::new(bytes), None, None, Selection::new()).await?;
//! let mut samples = Box::pin(group_events_stream(files.into_events(), reraise_exception()));
//!
//! let first = samples.next().await.expect("a sample")?;
//! assert_eq!(first.key(), Some("10"));
//! # Ok::<(), webdataset_core::Error>(())
//! # }).unwrap();
//! ```
//!
//! The whole pipeline above this layer — decoding, shuffling, batching — is
//! unchanged, because those are computations over samples rather than I/O.

use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use futures_io::AsyncRead;
use futures_util::io::{AsyncReadExt, BufReader};
use futures_util::stream::{self, StreamExt};
use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{Action, HandlerRef};
use webdataset_core::sample::Sample;

use crate::reader::{FileEvent, Grouped, Grouper, Selection, TarFile};
use crate::sync::{BLOCK, RawEntry, header_name, parse_size, pax_path, verify_checksum};

/// A stream of samples, boxed so it can be named and stored.
pub type SampleStream = Pin<Box<dyn Stream<Item = Result<Sample>> + Send>>;

/// A readable byte source an archive can be parsed from.
pub trait Source: AsyncRead + Send + Unpin {}
impl<T: AsyncRead + Send + Unpin> Source for T {}

/// The regular files of a tar archive, read in order and without blocking.
///
/// Parsing is the same as [`TarEntries`](crate::sync::TarEntries); only the
/// reads differ.
pub struct AsyncTarEntries<R> {
    reader: R,
    done: bool,
    /// A name supplied by a preceding GNU `L` record.
    long_name: Option<String>,
    /// A path supplied by a preceding PAX `x` record.
    pax_path: Option<String>,
}

impl<R: AsyncRead + Unpin> AsyncTarEntries<R> {
    /// Parse `reader` as an uncompressed tar archive.
    pub fn new(reader: R) -> AsyncTarEntries<R> {
        AsyncTarEntries { reader, done: false, long_name: None, pax_path: None }
    }

    /// Read the next regular file, or `None` at the end of the archive.
    ///
    /// Once this reports an error the archive is finished: a tar reader cannot
    /// resynchronise part way through a stream.
    pub async fn next_entry(&mut self) -> Result<Option<RawEntry>> {
        if self.done {
            return Ok(None);
        }
        match self.advance().await {
            Ok(Some(entry)) => Ok(Some(entry)),
            Ok(None) => {
                self.done = true;
                Ok(None)
            }
            Err(e) => {
                self.done = true;
                Err(e)
            }
        }
    }

    async fn advance(&mut self) -> Result<Option<RawEntry>> {
        loop {
            let Some(header) = self.read_header().await? else {
                return Ok(None);
            };

            let size = parse_size(&header)?;
            let typeflag = header[crate::sync::TYPEFLAG];

            match typeflag {
                // GNU long name: the payload names the entry that follows.
                b'L' => {
                    let payload = self.read_payload(size).await?;
                    self.long_name = Some(crate::sync::trim_nul(&payload)?);
                    continue;
                }
                // GNU long link name: irrelevant here, but must be consumed.
                b'K' => {
                    self.read_payload(size).await?;
                    continue;
                }
                // PAX extended header: may override the following entry's path.
                b'x' | b'X' => {
                    let records = self.read_payload(size).await?;
                    if let Some(path) = pax_path(&records) {
                        self.pax_path = Some(path);
                    }
                    continue;
                }
                // PAX global header: applies to the archive; skipped.
                b'g' => {
                    self.read_payload(size).await?;
                    continue;
                }
                // Regular files, in both the modern and the historic spelling.
                b'0' | b'\0' | b'7' => {}
                // Directories, links, devices and fifos hold no sample data.
                _ => {
                    self.read_payload(size).await?;
                    self.long_name = None;
                    self.pax_path = None;
                    continue;
                }
            }

            let name = match self.pax_path.take().or_else(|| self.long_name.take()) {
                Some(name) => name,
                None => header_name(&header)?,
            };
            return Ok(Some(RawEntry { name, data: self.read_payload(size).await? }));
        }
    }

    /// Read one header block, returning `None` at the end-of-archive marker.
    async fn read_header(&mut self) -> Result<Option<[u8; BLOCK]>> {
        let mut header = [0u8; BLOCK];
        match self.fill(&mut header).await? {
            Filled::Eof => return Ok(None),
            Filled::Partial(n) => {
                return Err(Error::format(format!("tar archive ends mid-header, {n} of {BLOCK} bytes")));
            }
            Filled::Complete => {}
        }
        // A run of zero bytes marks the end of the archive.
        if header.iter().all(|b| *b == 0) {
            return Ok(None);
        }
        verify_checksum(&header)?;
        Ok(Some(header))
    }

    /// Read `size` bytes of payload plus the padding up to the next block.
    async fn read_payload(&mut self, size: u64) -> Result<Vec<u8>> {
        let size = usize::try_from(size).map_err(|_| Error::format("tar entry is too large for this platform"))?;
        // Grow with the bytes that actually arrive rather than reserving what
        // the header claims. A shard is untrusted input, and a crafted header
        // can name a size no machine can allocate; reserving that up front
        // aborts the process, which no error handler can catch. Growing as the
        // data arrives turns the same input into an ordinary error.
        let mut data = Vec::with_capacity(size.min(crate::sync::MAX_PREALLOC));
        while data.len() < size {
            let start = data.len();
            data.resize(start + crate::sync::CHUNK.min(size - start), 0);
            if !matches!(self.fill(&mut data[start..]).await?, Filled::Complete) {
                return Err(Error::format(format!("tar archive ends mid-entry, expected {size} bytes")));
            }
        }
        let padding = (BLOCK - size % BLOCK) % BLOCK;
        if padding > 0 {
            let mut skip = [0u8; BLOCK];
            if !matches!(self.fill(&mut skip[..padding]).await?, Filled::Complete) {
                return Err(Error::format("tar archive ends mid-padding"));
            }
        }
        Ok(data)
    }

    /// Read exactly `buf.len()` bytes, distinguishing a clean end of stream.
    async fn fill(&mut self, buf: &mut [u8]) -> Result<Filled> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.reader.read(&mut buf[filled..]).await? {
                0 if filled == 0 => return Ok(Filled::Eof),
                0 => return Ok(Filled::Partial(filled)),
                n => filled += n,
            }
        }
        Ok(Filled::Complete)
    }

    /// Present the entries as a [`Stream`].
    pub fn into_stream(self) -> impl Stream<Item = Result<RawEntry>>
    where
        R: Unpin,
    {
        stream::unfold(Some(self), |state| async move {
            let mut entries = state?;
            match entries.next_entry().await {
                Ok(Some(entry)) => Some((Ok(entry), Some(entries))),
                Ok(None) => None,
                Err(e) => Some((Err(e), None)),
            }
        })
    }
}

/// How much of a requested read was satisfied.
enum Filled {
    /// Every byte arrived.
    Complete,
    /// The stream ended before any byte arrived.
    Eof,
    /// The stream ended part way through.
    Partial(usize),
}

/// The selected files of one archive, decompressed and parsed as they arrive.
pub struct AsyncShardFiles {
    entries: AsyncTarEntries<Box<dyn Source>>,
    url: Option<String>,
    local_path: Option<String>,
    selection: Selection,
}

impl AsyncShardFiles {
    /// Parse `stream`, tagging every file with `url` and `local_path`.
    ///
    /// Compression is detected from the leading bytes, as in the blocking
    /// reader, which is why this is a future: the sniff needs a read.
    pub async fn new(
        stream: impl Source + 'static,
        url: Option<String>,
        local_path: Option<String>,
        selection: Selection,
    ) -> Result<AsyncShardFiles> {
        let (_, decoded) = decompress(Box::new(stream)).await?;
        Ok(AsyncShardFiles { entries: AsyncTarEntries::new(decoded), url, local_path, selection })
    }

    /// Read the next selected file, or `None` at the end of the archive.
    pub async fn next_file(&mut self) -> Result<Option<TarFile>> {
        loop {
            let Some(entry) = self.entries.next_entry().await.map_err(|e| e.context(describe(&self.url)))? else {
                return Ok(None);
            };
            let Some(name) = self.selection.apply(&entry.name) else {
                continue;
            };
            return Ok(Some(TarFile {
                name,
                data: Bytes::from(entry.data),
                url: self.url.clone(),
                local_path: self.local_path.clone(),
            }));
        }
    }

    /// Present the files as a [`Stream`].
    pub fn into_stream(self) -> impl Stream<Item = Result<TarFile>> {
        stream::unfold(Some(self), |state| async move {
            let mut files = state?;
            match files.next_file().await {
                Ok(Some(file)) => Some((Ok(file), Some(files))),
                Ok(None) => None,
                Err(e) => Some((Err(e), None)),
            }
        })
    }

    /// Present the files as a stream of events ending in a shard boundary.
    ///
    /// The boundary is emitted even when the shard fails part way through, so a
    /// truncated shard can never merge its trailing files into the next one.
    pub fn into_events(self) -> impl Stream<Item = Result<FileEvent>> {
        self.into_stream()
            .map(|item| item.map(FileEvent::File))
            .chain(stream::once(async { Ok(FileEvent::EndOfShard) }))
    }
}

fn describe(url: &Option<String>) -> String {
    match url {
        Some(url) => format!("{url}: "),
        None => String::new(),
    }
}

/// Group a stream of [`FileEvent`]s into samples, honouring shard boundaries.
///
/// The grouping rule is shared with the blocking reader; see [`Grouper`].
pub fn group_events_stream<S>(events: S, handler: HandlerRef) -> impl Stream<Item = Result<Sample>>
where
    S: Stream<Item = Result<FileEvent>>,
{
    /// What the fold carries between events.
    struct State<S> {
        events: Pin<Box<S>>,
        grouper: Grouper,
        done: bool,
    }

    let state = State { events: Box::pin(events), grouper: Grouper::new(handler), done: false };

    stream::unfold(Some(state), |carried| async move {
        let mut state = carried?;
        loop {
            if state.done {
                // Emit whatever sample was still being assembled, then stop.
                return state.grouper.finish().map(|sample| (Ok(sample), None));
            }
            let event = match state.events.next().await {
                Some(Ok(event)) => event,
                Some(Err(e)) => match state.grouper.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        state.done = true;
                        continue;
                    }
                    Action::Reraise => {
                        state.done = true;
                        return Some((Err(e), Some(state)));
                    }
                },
                None => {
                    state.done = true;
                    continue;
                }
            };
            match state.grouper.push(event) {
                Grouped::Sample(sample) => return Some((Ok(sample), Some(state))),
                Grouped::Pending => continue,
                Grouped::Failed(e) => return Some((Err(e), Some(state))),
                Grouped::Stop => {
                    state.done = true;
                    continue;
                }
            }
        }
    })
}

/// A stream of file events, boxed so the two arms below have one type.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<FileEvent>> + Send>>;

/// Read one shard from `stream` and yield its samples.
///
/// Failing to open the shard is reported through `handler` just like failing to
/// read one, so a caller never has to special-case the two.
pub fn shard_samples(
    stream: impl Source + 'static,
    url: Option<String>,
    local_path: Option<String>,
    selection: Selection,
    handler: HandlerRef,
) -> SampleStream {
    let opening = stream::once(async move { AsyncShardFiles::new(stream, url, local_path, selection).await });
    let events = opening.flat_map(|opened| -> EventStream {
        match opened {
            Ok(files) => Box::pin(files.into_events()),
            Err(e) => Box::pin(stream::once(async move { Err(e) })),
        }
    });
    Box::pin(group_events_stream(events, handler))
}

/// Wrap `stream` in the decoder its magic bytes call for.
///
/// The container is detected from the leading bytes rather than from a file
/// name, so a shard read from a socket decompresses just as one read from disk.
pub async fn decompress(mut stream: Box<dyn Source>) -> Result<(webdataset_io::cache::FileType, Box<dyn Source>)> {
    use webdataset_io::cache::{FileType, sniff};

    let mut prefix = vec![0u8; crate::compress::SNIFF_LEN];
    let mut filled = 0;
    while filled < prefix.len() {
        match stream.read(&mut prefix[filled..]).await? {
            0 => break,
            n => filled += n,
        }
    }
    prefix.truncate(filled);

    let kind = sniff(&prefix);
    let combined = Prefixed { prefix: futures_util::io::Cursor::new(prefix), rest: stream };
    let decoded: Box<dyn Source> = match kind {
        FileType::Gzip => {
            let mut decoder = async_compression::futures::bufread::GzipDecoder::new(BufReader::new(combined));
            // Shards are sometimes concatenated gzip members.
            decoder.multiple_members(true);
            Box::new(decoder)
        }
        #[cfg(feature = "zstd")]
        FileType::Zstd => {
            let mut decoder = async_compression::futures::bufread::ZstdDecoder::new(BufReader::new(combined));
            decoder.multiple_members(true);
            Box::new(decoder)
        }
        #[cfg(feature = "bzip2")]
        FileType::Bzip2 => {
            let mut decoder = async_compression::futures::bufread::BzDecoder::new(BufReader::new(combined));
            decoder.multiple_members(true);
            Box::new(decoder)
        }
        #[cfg(feature = "xz")]
        FileType::Xz => {
            let mut decoder = async_compression::futures::bufread::XzDecoder::new(BufReader::new(combined));
            decoder.multiple_members(true);
            Box::new(decoder)
        }
        FileType::Tar | FileType::Unknown => Box::new(combined),
        #[allow(unreachable_patterns)]
        other => {
            return Err(Error::unsupported(format!(
                "{} shards; rebuild webdataset-shard with the matching feature enabled",
                other.name()
            )));
        }
    };
    Ok((kind, decoded))
}

/// A reader that replays a prefix already consumed from an inner stream.
struct Prefixed {
    prefix: futures_util::io::Cursor<Vec<u8>>,
    rest: Box<dyn Source>,
}

impl AsyncRead for Prefixed {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut [u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        use std::task::Poll;
        let this = &mut *self;
        match Pin::new(&mut this.prefix).poll_read(cx, buf) {
            Poll::Ready(Ok(0)) => Pin::new(&mut this.rest).poll_read(cx, buf),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_executor::block_on;
    use futures_util::io::Cursor;
    use futures_util::stream::TryStreamExt;
    use webdataset_core::handlers::{ignore_and_continue, reraise_exception};

    fn shard(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name);
        std::fs::read(path).expect("reading the shard")
    }

    async fn samples(name: &str) -> Vec<Sample> {
        let files = AsyncShardFiles::new(Cursor::new(shard(name)), Some(name.to_string()), None, Selection::new())
            .await
            .expect("opening the shard");
        group_events_stream(files.into_events(), reraise_exception())
            .try_collect::<Vec<_>>()
            .await
            .expect("reading the samples")
    }

    #[test]
    fn reads_an_uncompressed_shard() {
        block_on(async {
            let samples = samples("mpdata.tar").await;
            assert_eq!(samples.len(), 100);
            assert_eq!(samples[0].key(), Some("000000"));
            assert!(samples[0].contains_key("mp"));
        });
    }

    #[test]
    fn decompresses_a_gzipped_shard() {
        block_on(async {
            let samples = samples("imagenet-000000.tgz").await;
            assert_eq!(samples.len(), 47);
            let mut fields = samples[0].field_names();
            fields.sort();
            assert_eq!(fields, ["cls", "png", "wnid", "xml"]);
            assert_eq!(samples[0].url(), Some("imagenet-000000.tgz"));
        });
    }

    #[test]
    fn matches_the_blocking_reader_exactly() {
        for name in ["sample.tgz", "mpdata.tar", "tendata.tar", "testgz.tar", "ixtest.tar", "compressed.tar"] {
            let blocking: Vec<Sample> = crate::group_by_keys(
                crate::reader::ShardFiles::new(
                    Box::new(std::io::Cursor::new(shard(name))),
                    None,
                    None,
                    Selection::new(),
                )
                .expect("opening"),
            )
            .map(|s| s.expect("a sample"))
            .collect();

            let asynchronous = block_on(async {
                let files = AsyncShardFiles::new(Cursor::new(shard(name)), None, None, Selection::new())
                    .await
                    .expect("opening");
                group_events_stream(files.into_events(), reraise_exception())
                    .try_collect::<Vec<_>>()
                    .await
                    .expect("reading")
            });

            assert_eq!(asynchronous, blocking, "{name} read differently");
            assert!(!blocking.is_empty(), "{name} should not be empty");
        }
    }

    #[test]
    fn applies_selection_and_renaming() {
        block_on(async {
            let selection = Selection::new().select(|name| name.ends_with(".cls"));
            let files =
                AsyncShardFiles::new(Cursor::new(shard("sample.tgz")), None, None, selection).await.expect("opening");
            let samples: Vec<Sample> =
                group_events_stream(files.into_events(), reraise_exception()).try_collect().await.expect("reading");
            assert_eq!(samples.len(), 90);
            assert!(samples.iter().all(|s| s.field_names() == ["cls"]));
        });
    }

    #[test]
    fn reports_a_corrupt_archive() {
        block_on(async {
            let files = AsyncShardFiles::new(Cursor::new(vec![b'x'; 4096]), None, None, Selection::new())
                .await
                .expect("opening succeeds; parsing is what fails");
            let outcome: Vec<Result<Sample>> =
                group_events_stream(files.into_events(), reraise_exception()).collect().await;
            assert!(outcome.iter().any(Result::is_err), "a bad checksum should be reported");
        });
    }

    #[test]
    fn a_truncated_shard_can_be_skipped() {
        block_on(async {
            let mut bytes = shard("mpdata.tar");
            bytes.truncate(5000);
            let files = AsyncShardFiles::new(Cursor::new(bytes), None, None, Selection::new()).await.expect("opening");
            let samples: Vec<Sample> = group_events_stream(files.into_events(), ignore_and_continue())
                .try_collect()
                .await
                .expect("the handler absorbs the truncation");
            assert!(!samples.is_empty());
            assert!(samples.len() < 100);
        });
    }

    #[test]
    fn shard_samples_reads_a_whole_shard() {
        block_on(async {
            let samples: Vec<Sample> = shard_samples(
                Cursor::new(shard("sample.tgz")),
                Some("mem://s.tar".into()),
                None,
                Selection::new(),
                reraise_exception(),
            )
            .try_collect()
            .await
            .expect("reading");
            assert_eq!(samples.len(), 90);
            assert_eq!(samples[0].url(), Some("mem://s.tar"));
        });
    }

    #[test]
    fn reads_entries_one_at_a_time() {
        block_on(async {
            let mut entries = AsyncTarEntries::new(Cursor::new(shard("mpdata.tar")));
            let first = entries.next_entry().await.expect("reading").expect("an entry");
            assert_eq!(first.name, "000000.mp");
            assert!(!first.data.is_empty());

            let mut count = 1;
            while entries.next_entry().await.expect("reading").is_some() {
                count += 1;
            }
            assert_eq!(count, 100);
        });
    }

    #[test]
    fn survives_a_stream_that_dribbles_bytes() {
        /// Yields at most one byte per poll, the worst case for a parser.
        struct Dribble(Cursor<Vec<u8>>);

        impl AsyncRead for Dribble {
            fn poll_read(
                mut self: Pin<&mut Self>,
                cx: &mut std::task::Context<'_>,
                buf: &mut [u8],
            ) -> std::task::Poll<std::io::Result<usize>> {
                let take = buf.len().min(1);
                Pin::new(&mut self.0).poll_read(cx, &mut buf[..take])
            }
        }

        block_on(async {
            let files =
                AsyncShardFiles::new(Dribble(Cursor::new(shard("compressed.tar"))), None, None, Selection::new())
                    .await
                    .expect("opening");
            let samples: Vec<Sample> =
                group_events_stream(files.into_events(), reraise_exception()).try_collect().await.expect("reading");
            assert_eq!(samples.len(), 3);
        });
    }
}
