//! Reading samples out of tar archives.
//!
//! The path from bytes to samples has three steps, matching the Python
//! implementation's `tar_file_iterator`, `tar_file_expander`, and
//! `group_by_keys`:
//!
//! 1. [`TarFiles`] walks one archive and yields its regular files.
//! 2. [`expand`] concatenates several archives, tagging each file with the URL
//!    it came from and marking shard boundaries.
//! 3. [`group_by_keys`] collects consecutive files that share a basename into a
//!    [`Sample`].
//!
//! ```no_run
//! use webdataset_shard::{group_by_keys, open_shard};
//!
//! for sample in group_by_keys(open_shard("testdata/sample.tgz")?) {
//!     let sample = sample?;
//!     println!("{} has {:?}", sample.key().unwrap_or(""), sample.field_names());
//! }
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::io::Read;
use std::sync::Arc;

use bytes::Bytes;
use webdataset_core::error::{Error, Result};
use webdataset_core::handlers::{Action, HandlerRef, reraise_exception};
use webdataset_core::sample::{LOCAL_PATH, Sample, URL};
use webdataset_core::utils::{base_plus_ext, is_shard_metadata};
use webdataset_core::value::Value;
use webdataset_io::gopen::gopen;

use crate::compress::decompress;
use crate::prelude::*;
use crate::sync::TarEntries;

/// How many files the background reader may run ahead of the consumer.
#[cfg(feature = "threads")]
const READAHEAD: usize = 4;

/// One regular file read out of an archive.
#[derive(Debug, Clone)]
pub struct TarFile {
    /// The member name, after any renaming.
    pub name: String,
    /// The file's contents.
    pub data: Bytes,
    /// The URL of the archive it came from.
    pub url: Option<String>,
    /// The local path of the archive, when it was read from disk.
    pub local_path: Option<String>,
}

/// An item in the flattened stream of archive contents.
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// A file from the current archive.
    File(TarFile),
    /// The current archive ended; a sample cannot span this boundary.
    EndOfShard,
}

/// Decides whether an archive member is read.
pub type SelectFn = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// Rewrites an archive member's name before it is grouped into a sample.
pub type RenameFn = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Chooses which archive members to read, and under what name.
#[derive(Clone, Default)]
pub struct Selection {
    select: Option<SelectFn>,
    rename: Option<RenameFn>,
    skip_metadata: bool,
}

impl std::fmt::Debug for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Selection")
            .field("select", &self.select.is_some())
            .field("rename", &self.rename.is_some())
            .field("skip_metadata", &self.skip_metadata)
            .finish()
    }
}

impl Selection {
    /// Read every file, skipping shard-level `__meta__` entries.
    pub fn new() -> Selection {
        Selection { select: None, rename: None, skip_metadata: true }
    }

    /// Keep only the members for which `predicate` returns true.
    ///
    /// The predicate sees the name *after* renaming, as in the Python API.
    pub fn select(mut self, predicate: impl Fn(&str) -> bool + Send + Sync + 'static) -> Selection {
        self.select = Some(Arc::new(predicate));
        self
    }

    /// Rewrite member names before they are grouped into samples.
    pub fn rename(mut self, rename: impl Fn(&str) -> String + Send + Sync + 'static) -> Selection {
        self.rename = Some(Arc::new(rename));
        self
    }

    /// Whether to drop shard-level metadata entries such as `__index__`.
    pub fn skip_metadata(mut self, skip: bool) -> Selection {
        self.skip_metadata = skip;
        self
    }

    /// Apply the renaming and selection rules, returning the name to use.
    pub(crate) fn apply(&self, name: &str) -> Option<String> {
        if self.skip_metadata && is_shard_metadata(name) {
            return None;
        }
        let name = match &self.rename {
            Some(rename) => rename(name),
            None => name.to_string(),
        };
        match &self.select {
            Some(select) if !select(&name) => None,
            _ => Some(name),
        }
    }
}

/// The selected files of one archive, parsed as they are read.
///
/// This is the whole reader: [`TarFiles`] is either this iterator used directly
/// or the same iterator driven from a background thread.
pub struct ShardFiles {
    entries: TarEntries<Box<dyn Read + Send>>,
    url: Option<String>,
    local_path: Option<String>,
    selection: Selection,
}

impl ShardFiles {
    /// Parse `stream`, tagging every file with `url` and `local_path`.
    pub fn new(
        stream: Box<dyn Read + Send>,
        url: Option<String>,
        local_path: Option<String>,
        selection: Selection,
    ) -> Result<ShardFiles> {
        let (_, decoded) = decompress(stream)?;
        Ok(ShardFiles { entries: TarEntries::new(decoded), url, local_path, selection })
    }
}

impl Iterator for ShardFiles {
    type Item = Result<TarFile>;

    fn next(&mut self) -> Option<Result<TarFile>> {
        loop {
            match self.entries.next()? {
                Ok(entry) => {
                    let Some(name) = self.selection.apply(&entry.name) else {
                        continue;
                    };
                    return Some(Ok(TarFile {
                        name,
                        data: Bytes::from(entry.data),
                        url: self.url.clone(),
                        local_path: self.local_path.clone(),
                    }));
                }
                Err(e) => return Some(Err(e.context(describe(&self.url)))),
            }
        }
    }
}

/// The files of a single archive, read in order.
///
/// With the `threads` feature — on by default — parsing runs on a background
/// thread that stays a few files ahead of the consumer, so archive decoding
/// overlaps with whatever the consumer is doing. The thread stops on its own
/// when this iterator is dropped. Without the feature, or on targets that have
/// no threads such as WebAssembly, the same parser runs inline.
pub struct TarFiles {
    inner: Inner,
    done: bool,
}

enum Inner {
    #[cfg(not(feature = "threads"))]
    Inline(ShardFiles),
    #[cfg(feature = "threads")]
    Threaded(std::sync::mpsc::Receiver<Result<TarFile>>),
}

impl TarFiles {
    /// Read `stream` as an archive, decompressing it if necessary.
    pub fn new(stream: Box<dyn Read + Send>) -> Result<TarFiles> {
        TarFiles::with_options(stream, None, None, Selection::new())
    }

    /// Read `stream`, recording `url` and `local_path` on every file.
    pub fn with_options(
        stream: Box<dyn Read + Send>,
        url: Option<String>,
        local_path: Option<String>,
        selection: Selection,
    ) -> Result<TarFiles> {
        let files = ShardFiles::new(stream, url, local_path, selection)?;

        #[cfg(feature = "threads")]
        {
            let (sender, receiver) = std::sync::mpsc::sync_channel(READAHEAD);
            std::thread::Builder::new()
                .name("webdataset-shard".into())
                .spawn(move || {
                    for item in files {
                        // A failed send means the consumer went away.
                        let failed = item.is_err();
                        if sender.send(item).is_err() || failed {
                            return;
                        }
                    }
                })
                .map_err(|e| Error::Io(e).context("spawning the archive reader"))?;
            Ok(TarFiles { inner: Inner::Threaded(receiver), done: false })
        }

        #[cfg(not(feature = "threads"))]
        Ok(TarFiles { inner: Inner::Inline(files), done: false })
    }

    /// Open the archive at `url` through [`gopen`].
    pub fn open(url: &str) -> Result<TarFiles> {
        Self::open_with(url, Selection::new())
    }

    /// Open the archive at `url`, applying `selection`.
    pub fn open_with(url: &str, selection: Selection) -> Result<TarFiles> {
        let stream = gopen(url)?;
        let local = webdataset_io::url::is_local(url).then(|| webdataset_io::url::to_local_path(url));
        TarFiles::with_options(Box::new(FetchReader(stream)), Some(url.to_string()), local, selection)
    }
}

impl Iterator for TarFiles {
    type Item = Result<TarFile>;

    fn next(&mut self) -> Option<Result<TarFile>> {
        if self.done {
            return None;
        }
        let item = match &mut self.inner {
            #[cfg(not(feature = "threads"))]
            Inner::Inline(files) => files.next(),
            #[cfg(feature = "threads")]
            // The sender being gone means the archive is finished.
            Inner::Threaded(receiver) => receiver.recv().ok(),
        };
        if item.as_ref().is_none_or(Result::is_err) {
            self.done = true;
        }
        item
    }
}

/// Adapts a [`Fetch`](webdataset_io::gopen::Fetch) into a plain reader.
struct FetchReader(Box<dyn webdataset_io::gopen::Fetch>);

impl Read for FetchReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

fn describe(url: &Option<String>) -> String {
    match url {
        Some(url) => format!("{url}: "),
        None => String::new(),
    }
}

/// Open one shard and stream its files, without shard boundary markers.
pub fn open_shard(url: &str) -> Result<TarFiles> {
    TarFiles::open(url)
}

/// Flatten several archives into one stream of files and shard boundaries.
///
/// `sources` yields `(url, stream)` pairs. Errors opening or reading an archive
/// are passed to `handler`, which decides whether to skip the shard, stop, or
/// report the error downstream — the same contract the Python
/// `tar_file_expander` has.
pub fn expand<I>(sources: I, selection: Selection, handler: HandlerRef) -> Expand<I>
where
    I: Iterator<Item = Result<ShardSource>>,
{
    Expand { sources, selection, handler, current: None, pending_boundary: false, done: false }
}

/// An opened shard: its URL and the bytes behind it.
pub struct ShardSource {
    /// The URL the shard was named by.
    pub url: String,
    /// Where the shard lives on disk, if it does.
    pub local_path: Option<String>,
    /// The shard's bytes.
    pub stream: Box<dyn Read + Send>,
}

impl std::fmt::Debug for ShardSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardSource").field("url", &self.url).field("local_path", &self.local_path).finish()
    }
}

/// The iterator returned by [`expand`].
pub struct Expand<I> {
    sources: I,
    selection: Selection,
    handler: HandlerRef,
    current: Option<TarFiles>,
    pending_boundary: bool,
    done: bool,
}

impl<I> Iterator for Expand<I>
where
    I: Iterator<Item = Result<ShardSource>>,
{
    type Item = Result<FileEvent>;

    fn next(&mut self) -> Option<Result<FileEvent>> {
        loop {
            // A boundary is emitted after every shard, including one that
            // failed part way through, so a truncated shard can never merge
            // its trailing files into the next shard's first sample.
            if self.pending_boundary {
                self.pending_boundary = false;
                return Some(Ok(FileEvent::EndOfShard));
            }
            if self.done {
                return None;
            }

            if let Some(files) = self.current.as_mut() {
                match files.next() {
                    Some(Ok(file)) => return Some(Ok(FileEvent::File(file))),
                    Some(Err(e)) => {
                        self.current = None;
                        self.pending_boundary = true;
                        match self.handler.handle(&e) {
                            Action::Continue => continue,
                            Action::Stop => {
                                self.done = true;
                                continue;
                            }
                            Action::Reraise => return Some(Err(e)),
                        }
                    }
                    None => {
                        self.current = None;
                        self.pending_boundary = true;
                        continue;
                    }
                }
            }

            let source = match self.sources.next() {
                Some(Ok(source)) => source,
                Some(Err(e)) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        return None;
                    }
                    Action::Reraise => return Some(Err(e)),
                },
                None => {
                    self.done = true;
                    return None;
                }
            };

            let url = source.url.clone();
            let opened =
                TarFiles::with_options(source.stream, Some(source.url), source.local_path, self.selection.clone());
            match opened {
                Ok(files) => self.current = Some(files),
                Err(e) => {
                    let e = e.context(url);
                    match self.handler.handle(&e) {
                        Action::Continue => continue,
                        Action::Stop => {
                            self.done = true;
                            return None;
                        }
                        Action::Reraise => return Some(Err(e)),
                    }
                }
            }
        }
    }
}

/// Group consecutive files that share a basename into samples.
///
/// Use this when every file comes from the same archive; [`group_events`]
/// additionally understands shard boundaries.
pub fn group_by_keys<I>(files: I) -> GroupByKeys<AsEvents<I::IntoIter>>
where
    I: IntoIterator<Item = Result<TarFile>>,
{
    group_events(AsEvents(files.into_iter()), reraise_exception())
}

/// Lifts a stream of files into a stream of [`FileEvent`]s.
pub struct AsEvents<I>(I);

impl<I: Iterator<Item = Result<TarFile>>> Iterator for AsEvents<I> {
    type Item = Result<FileEvent>;

    fn next(&mut self) -> Option<Result<FileEvent>> {
        self.0.next().map(|item| item.map(FileEvent::File))
    }
}

impl From<TarFile> for FileEvent {
    fn from(file: TarFile) -> FileEvent {
        FileEvent::File(file)
    }
}

/// Group a stream of [`FileEvent`]s into samples, honouring shard boundaries.
pub fn group_events<I>(files: I, handler: HandlerRef) -> GroupByKeys<I>
where
    I: Iterator<Item = Result<FileEvent>>,
{
    GroupByKeys { files, handler: handler.clone(), grouper: Grouper::new(handler), done: false }
}

/// What feeding one event to a [`Grouper`] produced.
#[derive(Debug)]
pub enum Grouped {
    /// A sample was completed and should be emitted.
    Sample(Sample),
    /// Nothing yet; feed the next event.
    Pending,
    /// An error the handler asked to be reported.
    Failed(Error),
    /// The handler asked for the stream to end.
    Stop,
}

/// Assembles consecutive files that share a basename into samples.
///
/// The grouping rule is the whole of the WebDataset format, so it lives in one
/// place and both the blocking and the asynchronous readers drive it. Feed it
/// [`FileEvent`]s with [`Grouper::push`] and take whatever is left at the end
/// with [`Grouper::finish`].
#[derive(Debug)]
pub struct Grouper {
    current: Option<Sample>,
    handler: HandlerRef,
    lowercase: bool,
}

impl Grouper {
    /// A grouper that lowercases extensions, as the reference implementation does.
    pub fn new(handler: HandlerRef) -> Grouper {
        Grouper { current: None, handler, lowercase: true }
    }

    /// Whether to lowercase extensions.
    pub fn lowercase(mut self, lowercase: bool) -> Grouper {
        self.lowercase = lowercase;
        self
    }

    /// Feed one event.
    pub fn push(&mut self, event: FileEvent) -> Grouped {
        let file = match event {
            FileEvent::File(file) => file,
            // A sample never spans two shards.
            FileEvent::EndOfShard => {
                return match self.current.take().filter(Sample::is_valid) {
                    Some(sample) => Grouped::Sample(sample),
                    None => Grouped::Pending,
                };
            }
        };

        // A file with no extension at all belongs to no sample.
        let Some((prefix, suffix)) = base_plus_ext(&file.name) else {
            return Grouped::Pending;
        };
        let suffix = if self.lowercase { suffix.to_ascii_lowercase() } else { suffix.to_string() };

        let mut finished = None;
        if self.current.as_ref().and_then(Sample::key) != Some(prefix) {
            finished = self.current.take().filter(Sample::is_valid);
            let mut sample = Sample::with_key(prefix);
            if let Some(url) = &file.url {
                sample.insert(URL, Value::Text(url.clone()));
            }
            if let Some(path) = &file.local_path {
                sample.insert(LOCAL_PATH, Value::Text(path.clone()));
            }
            self.current = Some(sample);
        }

        let sample = self.current.as_mut().expect("just created");
        if sample.contains_key(&suffix) {
            let error = Error::DuplicateKey { key: suffix, sample_key: prefix.to_string() };
            match self.handler.handle(&error) {
                Action::Continue => {}
                Action::Stop => return Grouped::Stop,
                Action::Reraise => return Grouped::Failed(error),
            }
        } else {
            sample.insert(suffix, Value::Bytes(file.data));
        }

        match finished {
            Some(sample) => Grouped::Sample(sample),
            None => Grouped::Pending,
        }
    }

    /// Take the sample still being assembled, if it is usable.
    pub fn finish(&mut self) -> Option<Sample> {
        self.current.take().filter(Sample::is_valid)
    }

    /// How the handler wants an upstream error treated.
    pub fn handle(&self, error: &Error) -> Action {
        self.handler.handle(error)
    }
}

/// The iterator returned by [`group_by_keys`].
pub struct GroupByKeys<I> {
    files: I,
    handler: HandlerRef,
    grouper: Grouper,
    done: bool,
}

impl<I> GroupByKeys<I> {
    /// Whether to lowercase extensions; on by default, as in Python.
    pub fn lowercase(mut self, lowercase: bool) -> Self {
        self.grouper = self.grouper.lowercase(lowercase);
        self
    }
}

impl<I> Iterator for GroupByKeys<I>
where
    I: Iterator<Item = Result<FileEvent>>,
{
    type Item = Result<Sample>;

    fn next(&mut self) -> Option<Result<Sample>> {
        loop {
            if self.done {
                return self.grouper.finish().map(Ok);
            }
            let event = match self.files.next() {
                Some(Ok(event)) => event,
                Some(Err(e)) => match self.handler.handle(&e) {
                    Action::Continue => continue,
                    Action::Stop => {
                        self.done = true;
                        continue;
                    }
                    Action::Reraise => {
                        self.done = true;
                        return Some(Err(e));
                    }
                },
                None => {
                    self.done = true;
                    continue;
                }
            };

            match self.grouper.push(event) {
                Grouped::Sample(sample) => return Some(Ok(sample)),
                Grouped::Pending => continue,
                Grouped::Failed(e) => return Some(Err(e)),
                Grouped::Stop => {
                    self.done = true;
                    continue;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn testdata(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name)
    }

    fn files(name: &str) -> Vec<TarFile> {
        TarFiles::open(testdata(name).to_str().unwrap()).unwrap().map(|f| f.unwrap()).collect()
    }

    fn samples(name: &str) -> Vec<Sample> {
        group_by_keys(TarFiles::open(testdata(name).to_str().unwrap()).unwrap()).map(|s| s.unwrap()).collect()
    }

    #[test]
    fn reads_an_uncompressed_archive() {
        let files = files("mpdata.tar");
        assert_eq!(files.len(), 100);
        assert_eq!(files[0].name, "000000.mp");
        assert!(!files[0].data.is_empty());
        assert!(files[0].url.as_deref().unwrap().ends_with("mpdata.tar"));
    }

    #[test]
    fn transparently_decompresses_a_tgz() {
        let samples = samples("imagenet-000000.tgz");
        assert_eq!(samples.len(), 47);
        let first = &samples[0];
        assert_eq!(first.key(), Some("10"));
        let mut names = first.field_names();
        names.sort();
        assert_eq!(names, ["cls", "png", "wnid", "xml"]);
    }

    #[test]
    fn groups_files_that_share_a_basename() {
        let samples = samples("sample.tgz");
        assert!(!samples.is_empty());
        for sample in &samples {
            assert!(sample.key().is_some());
            assert!(sample.contains_key("cls"), "{:?}", sample.field_names());
            assert!(sample.contains_key("png"), "{:?}", sample.field_names());
            assert!(sample.url().is_some());
        }
    }

    #[test]
    fn keeps_multi_part_extensions_intact() {
        let samples = samples("testgz.tar");
        assert!(!samples.is_empty());
        assert!(samples[0].contains_key("txt.gz"), "{:?}", samples[0].field_names());
    }

    #[test]
    fn lowercases_extensions() {
        let file = TarFile { name: "key.PNG".into(), data: Bytes::from_static(b"x"), url: None, local_path: None };
        let grouped: Vec<Sample> = group_by_keys(vec![Ok(file)]).map(|s| s.unwrap()).collect();
        assert_eq!(grouped[0].field_names(), ["png"]);
    }

    #[test]
    fn never_lets_a_sample_span_two_shards() {
        let make =
            |name: &str| TarFile { name: name.into(), data: Bytes::from_static(b"x"), url: None, local_path: None };
        let events =
            vec![Ok(FileEvent::File(make("a.cls"))), Ok(FileEvent::EndOfShard), Ok(FileEvent::File(make("a.png")))];
        let grouped: Vec<Sample> = group_events(events.into_iter(), reraise_exception()).map(|s| s.unwrap()).collect();

        assert_eq!(grouped.len(), 2, "the boundary must split the two files apart");
        assert_eq!(grouped[0].field_names(), ["cls"]);
        assert_eq!(grouped[1].field_names(), ["png"]);
    }

    #[test]
    fn reports_duplicate_fields() {
        let make =
            |name: &str| Ok(TarFile { name: name.into(), data: Bytes::from_static(b"x"), url: None, local_path: None });
        let mut grouped = group_by_keys(vec![make("a.png"), make("a.png")]);
        let err = grouped.next().unwrap().unwrap_err();
        assert!(matches!(err, Error::DuplicateKey { .. }), "{err}");
    }

    #[test]
    fn applies_selection_and_renaming() {
        let selection = Selection::new().rename(|n| n.replace(".png", ".image")).select(|n| n.ends_with(".image"));
        let files: Vec<TarFile> = TarFiles::open_with(testdata("sample.tgz").to_str().unwrap(), selection)
            .unwrap()
            .map(|f| f.unwrap())
            .collect();
        assert!(!files.is_empty());
        assert!(files.iter().all(|f| f.name.ends_with(".image")), "{:?}", &files[..2]);
    }

    #[test]
    fn skips_shard_metadata_entries() {
        let selection = Selection::new();
        assert_eq!(selection.apply("__index__"), None);
        assert_eq!(selection.apply("a.png").as_deref(), Some("a.png"));
        assert_eq!(Selection::new().skip_metadata(false).apply("__index__").as_deref(), Some("__index__"));
    }

    #[test]
    fn reports_a_corrupt_archive() {
        let junk = vec![b'x'; 4096];
        let files = TarFiles::new(Box::new(std::io::Cursor::new(junk))).unwrap();
        let outcome: Vec<Result<TarFile>> = files.collect();
        assert!(outcome.iter().any(|f| f.is_err()), "a corrupt archive should surface an error");
    }

    #[test]
    fn allows_dropping_a_shard_part_way_through() {
        let path = testdata("mpdata.tar");
        let url = path.to_str().unwrap();

        // Read two files, then abandon the archive: the background reader must
        // not keep the next open from making progress.
        {
            let mut files = TarFiles::open(url).unwrap();
            assert!(files.next().is_some());
            assert!(files.next().is_some());
        }
        let again: Vec<TarFile> = TarFiles::open(url).unwrap().map(|f| f.unwrap()).collect();
        assert_eq!(again.len(), 100);
    }

    #[test]
    fn the_inline_reader_agrees_with_the_threaded_one() {
        let path = testdata("mpdata.tar");
        let open = || Box::new(std::fs::File::open(&path).unwrap()) as Box<dyn std::io::Read + Send>;

        let threaded: Vec<(String, usize)> = TarFiles::with_options(open(), None, None, Selection::new())
            .unwrap()
            .map(|f| {
                let f = f.unwrap();
                (f.name, f.data.len())
            })
            .collect();

        let inline: Vec<(String, usize)> = ShardFiles::new(open(), None, None, Selection::new())
            .unwrap()
            .map(|f| {
                let f = f.unwrap();
                (f.name, f.data.len())
            })
            .collect();

        assert_eq!(threaded, inline);
        assert_eq!(inline.len(), 100);
    }

    #[test]
    fn expands_several_shards_with_boundaries() {
        let sources = ["sample.tgz", "mpdata.tar"].map(|name| {
            let path = testdata(name);
            Ok(ShardSource {
                url: path.to_string_lossy().into_owned(),
                local_path: Some(path.to_string_lossy().into_owned()),
                stream: Box::new(std::fs::File::open(&path).unwrap()),
            })
        });
        let events: Vec<FileEvent> =
            expand(sources.into_iter(), Selection::new(), reraise_exception()).map(|e| e.unwrap()).collect();

        let boundaries = events.iter().filter(|e| matches!(e, FileEvent::EndOfShard)).count();
        assert_eq!(boundaries, 2, "one boundary per shard");
        assert!(matches!(events.last(), Some(FileEvent::EndOfShard)));
    }
}
