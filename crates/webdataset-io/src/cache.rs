//! Caching shards on local disk.
//!
//! Training over remote shards repeatedly is wasteful, so WebDataset can keep a
//! local copy of each shard it downloads. [`FileCache`] handles the download,
//! validates that the result really is an archive, and evicts old shards once
//! the cache grows past a budget.
//!
//! Downloads go to a process-unique temporary name and are renamed into place
//! only once complete, so several workers can populate the same cache directory
//! without ever observing a half-written shard.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use webdataset_core::error::{Error, Result};
use webdataset_core::utils::next_unique;

use crate::gopen::{Fetch, gopen};
use crate::url;

/// The cache directory used when none is configured.
pub const DEFAULT_CACHE_DIR: &str = "./_cache";

/// How much of a file to inspect when guessing its type.
const MAGIC_BYTES: usize = 512;

/// Derive a cache file name from a URL.
///
/// For ordinary schemes this is the last `ndir + 1` path components, so
/// `https://host/a/b/shard-000.tar` with `ndir = 0` caches as `shard-000.tar`.
/// Anything else (notably `pipe:` URLs) is percent-encoded whole and truncated
/// to the last 128 characters.
///
/// ```
/// use webdataset_io::cache::url_to_cache_name;
///
/// assert_eq!(url_to_cache_name("https://host/a/b/shard-000.tar", 0), "shard-000.tar");
/// assert_eq!(url_to_cache_name("https://host/a/b/shard-000.tar", 1), "b/shard-000.tar");
/// ```
pub fn url_to_cache_name(url: &str, ndir: usize) -> String {
    let scheme = url::scheme(url);
    let known = matches!(scheme, None | Some("file" | "http" | "https" | "ftp" | "ftps" | "gs" | "s3" | "ais"));
    if known {
        let path = url::path(url);
        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        let start = parts.len().saturating_sub(ndir + 1);
        return parts[start..].join("/");
    }
    let quoted = url::percent_encode(url, "_+{}*,-");
    let start = quoted.len().saturating_sub(128);
    quoted[start..].to_string()
}

/// Guess a file's type from its leading bytes.
///
/// Only the formats WebDataset shards actually use are recognised.
pub fn magic_filetype(path: &Path) -> Result<FileType> {
    let mut file = fs::File::open(path)?;
    let mut header = vec![0u8; MAGIC_BYTES];
    let mut filled = 0;
    while filled < header.len() {
        match file.read(&mut header[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    header.truncate(filled);
    Ok(sniff(&header))
}

/// Identify a stream from its first bytes.
pub fn sniff(header: &[u8]) -> FileType {
    if header.starts_with(&[0x1f, 0x8b]) {
        return FileType::Gzip;
    }
    if header.starts_with(b"BZh") {
        return FileType::Bzip2;
    }
    if header.starts_with(&[0xfd, b'7', b'z', b'X', b'Z', 0x00]) {
        return FileType::Xz;
    }
    if header.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        return FileType::Zstd;
    }
    if header.len() > 262 && &header[257..262] == b"ustar" {
        return FileType::Tar;
    }
    FileType::Unknown
}

/// The container formats a shard can arrive in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// An uncompressed POSIX tar archive, identified by its `ustar` magic.
    Tar,
    /// gzip, as `.tar.gz` and `.tgz` shards are.
    Gzip,
    /// bzip2.
    Bzip2,
    /// XZ.
    Xz,
    /// Zstandard.
    Zstd,
    /// None of the above. An old V7 tar has no magic, so this is still read as
    /// an archive rather than refused.
    Unknown,
}

impl FileType {
    /// Whether a shard in this format can be opened as a tar archive.
    pub fn is_archive(self) -> bool {
        !matches!(self, FileType::Unknown)
    }

    /// A human readable name.
    pub fn name(self) -> &'static str {
        match self {
            FileType::Tar => "POSIX tar archive",
            FileType::Gzip => "gzip compressed data",
            FileType::Bzip2 => "bzip2 compressed data",
            FileType::Xz => "XZ compressed data",
            FileType::Zstd => "Zstandard compressed data",
            FileType::Unknown => "data",
        }
    }
}

/// Whether the file at `path` looks like a shard.
pub fn check_tar_format(path: &Path) -> Result<bool> {
    Ok(magic_filetype(path)?.is_archive())
}

/// Copy `url` to `dest`, via a temporary file that is renamed on completion.
pub fn download(url: &str, dest: &Path) -> Result<u64> {
    let temp = dest.with_extension(format!("temp{}-{}", std::process::id(), next_unique()));
    let result = copy_to(url, &temp);
    match result {
        Ok(written) => {
            fs::rename(&temp, dest)?;
            Ok(written)
        }
        Err(e) => {
            fs::remove_file(&temp).ok();
            Err(e)
        }
    }
}

fn copy_to(url: &str, temp: &Path) -> Result<u64> {
    let mut stream = gopen(url)?;
    let mut file = std::io::BufWriter::new(fs::File::create(temp)?);
    let written = std::io::copy(&mut stream, &mut file)?;
    file.flush()?;
    stream.finish()?;
    Ok(written)
}

/// Evicts the least recently created files once a directory exceeds a budget.
#[derive(Debug)]
pub struct LruCleanup {
    directory: PathBuf,
    budget: u64,
    interval: Option<Duration>,
    last_run: std::sync::Mutex<Option<Instant>>,
}

impl LruCleanup {
    /// Keep `directory` below `budget` bytes, checking at most every `interval`.
    pub fn new(directory: impl Into<PathBuf>, budget: u64, interval: Option<Duration>) -> LruCleanup {
        LruCleanup { directory: directory.into(), budget, interval, last_run: std::sync::Mutex::new(None) }
    }

    /// Delete oldest-first until the directory fits in the budget.
    ///
    /// Files that vanish underneath us — another worker cleaning the same
    /// directory — are skipped rather than treated as errors.
    pub fn cleanup(&self) -> Result<u64> {
        if !self.directory.exists() {
            return Ok(0);
        }
        {
            let mut last = self.last_run.lock().expect("cleanup lock poisoned");
            if let (Some(interval), Some(at)) = (self.interval, *last) {
                if at.elapsed() < interval {
                    return Ok(0);
                }
            }
            *last = Some(Instant::now());
        }

        let mut entries = Vec::new();
        let mut total: u64 = 0;
        collect(&self.directory, &mut entries, &mut total)?;
        if total <= self.budget {
            return Ok(0);
        }

        // Oldest first, so the newest shards survive.
        entries.sort_by_key(|(_, _, created)| *created);

        let mut freed = 0u64;
        for (path, size, _) in entries {
            if total <= self.budget {
                break;
            }
            match fs::remove_file(&path) {
                Ok(()) => {
                    log::debug!("evicting {} ({size} bytes)", path.display());
                    total -= size.min(total);
                    freed += size;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(Error::Io(e).context(format!("removing {}", path.display()))),
            }
        }
        Ok(freed)
    }
}

/// Walk `directory`, collecting `(path, size, created)` and the total size.
fn collect(directory: &Path, out: &mut Vec<(PathBuf, u64, std::time::SystemTime)>, total: &mut u64) -> Result<()> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            collect(&path, out, total)?;
        } else {
            let created = meta.created().or_else(|_| meta.modified()).unwrap_or(std::time::UNIX_EPOCH);
            *total += meta.len();
            out.push((path, meta.len(), created));
        }
    }
    Ok(())
}

/// Downloads shards to a local directory and serves them from there.
pub struct FileCache {
    directory: PathBuf,
    name_of: Box<dyn Fn(&str) -> String + Send + Sync>,
    validate: bool,
    cleanup: Option<LruCleanup>,
    retries: usize,
}

impl std::fmt::Debug for FileCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileCache")
            .field("directory", &self.directory)
            .field("validate", &self.validate)
            .field("retries", &self.retries)
            .finish_non_exhaustive()
    }
}

impl FileCache {
    /// Cache shards under `directory`.
    pub fn new(directory: impl Into<PathBuf>) -> FileCache {
        FileCache {
            directory: directory.into(),
            name_of: Box::new(|url| url_to_cache_name(url, 0)),
            validate: true,
            cleanup: None,
            retries: 10,
        }
    }

    /// Cache under `WDS_CACHE` if it is set, else [`DEFAULT_CACHE_DIR`].
    pub fn from_env() -> FileCache {
        FileCache::new(std::env::var("WDS_CACHE").unwrap_or_else(|_| DEFAULT_CACHE_DIR.to_string()))
    }

    /// Choose cache file names with a custom function.
    pub fn with_naming(mut self, name_of: impl Fn(&str) -> String + Send + Sync + 'static) -> FileCache {
        self.name_of = Box::new(name_of);
        self
    }

    /// Whether to check that a downloaded shard really is an archive.
    pub fn with_validation(mut self, validate: bool) -> FileCache {
        self.validate = validate;
        self
    }

    /// Evict old shards once the cache exceeds `budget` bytes.
    pub fn with_budget(mut self, budget: u64, interval: Option<Duration>) -> FileCache {
        self.cleanup = Some(LruCleanup::new(self.directory.clone(), budget, interval));
        self
    }

    /// How many times to retry a failed download.
    pub fn with_retries(mut self, retries: usize) -> FileCache {
        self.retries = retries;
        self
    }

    /// The directory shards are cached in.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// The local path for `url`, downloading it first if necessary.
    ///
    /// Local URLs are passed through: there is nothing to gain from copying a
    /// file that is already on disk.
    pub fn get_file(&self, url: &str) -> Result<PathBuf> {
        if url::is_local(url) {
            return Ok(PathBuf::from(url::to_local_path(url)));
        }
        let name = (self.name_of)(url);
        let dest = self.directory.join(&name);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        if dest.exists() {
            return Ok(dest);
        }
        if let Some(cleanup) = &self.cleanup {
            cleanup.cleanup()?;
        }
        log::debug!("downloading {url} to {}", dest.display());
        download(url, &dest).map_err(|e| e.context(format!("caching {url}")))?;

        if self.validate && !check_tar_format(&dest)? {
            let kind = magic_filetype(&dest)?;
            fs::remove_file(&dest).ok();
            return Err(Error::format(format!("{url} is not an archive; it looks like {}", kind.name())));
        }
        Ok(dest)
    }

    /// Open `url`, reading from the cache and retrying transient failures.
    ///
    /// Returns the stream and the local path it was read from.
    pub fn open(&self, url: &str) -> Result<(Box<dyn Fetch>, PathBuf)> {
        let mut delay = Duration::from_secs(1);
        let mut last: Option<Error> = None;
        for attempt in 0..self.retries.max(1) {
            match self.get_file(url).and_then(|path| Ok((fs::File::open(&path)?, path))) {
                Ok((file, path)) => return Ok((Box::new(file), path)),
                Err(e) => {
                    log::debug!("attempt {} for {url} failed: {e}", attempt + 1);
                    last = Some(e);
                    if attempt + 1 < self.retries.max(1) {
                        std::thread::sleep(delay);
                        delay = delay.mul_f32(1.5);
                    }
                }
            }
        }
        Err(last.unwrap_or_else(|| Error::format(format!("could not open {url}"))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_cache_names() {
        assert_eq!(url_to_cache_name("https://host/a/b/s.tar", 0), "s.tar");
        assert_eq!(url_to_cache_name("https://host/a/b/s.tar", 1), "b/s.tar");
        assert_eq!(url_to_cache_name("local/s.tar", 0), "s.tar");
        let piped = url_to_cache_name("pipe:curl -s https://host/s.tar", 0);
        assert!(!piped.contains('/'), "{piped}");
        assert!(piped.len() <= 128);
    }

    #[test]
    fn sniffs_container_formats() {
        assert_eq!(sniff(&[0x1f, 0x8b, 0x08]), FileType::Gzip);
        assert_eq!(sniff(b"BZh9"), FileType::Bzip2);
        assert_eq!(sniff(&[0x28, 0xb5, 0x2f, 0xfd]), FileType::Zstd);
        assert_eq!(sniff(b"nope"), FileType::Unknown);

        let mut tar = vec![0u8; 512];
        tar[257..262].copy_from_slice(b"ustar");
        assert_eq!(sniff(&tar), FileType::Tar);
    }

    #[test]
    fn recognises_the_bundled_shards() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
        assert_eq!(magic_filetype(&root.join("imagenet-000000.tgz")).unwrap(), FileType::Gzip);
        assert_eq!(magic_filetype(&root.join("mpdata.tar")).unwrap(), FileType::Tar);
        assert!(check_tar_format(&root.join("mpdata.tar")).unwrap());
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn caches_a_shard_and_reuses_it() {
        let source = tempfile::tempdir().unwrap();
        let shard = source.path().join("shard-000.tar");
        let mut body = vec![0u8; 1024];
        body[257..262].copy_from_slice(b"ustar");
        fs::write(&shard, &body).unwrap();

        let cache_dir = tempfile::tempdir().unwrap();
        let cache = FileCache::new(cache_dir.path());
        let url = format!("pipe:cat {}", shard.display());

        let first = cache.get_file(&url).unwrap();
        assert!(first.exists());
        let modified = fs::metadata(&first).unwrap().modified().unwrap();

        let second = cache.get_file(&url).unwrap();
        assert_eq!(first, second);
        assert_eq!(fs::metadata(&second).unwrap().modified().unwrap(), modified, "cached shard was re-downloaded");
    }

    /// A scheme served entirely from memory, as a WebAssembly build would use.
    #[derive(Debug)]
    struct InMemory(&'static [u8]);

    impl crate::gopen::SchemeHandler for InMemory {
        fn open_read(&self, _url: &str) -> Result<Box<dyn crate::gopen::Fetch>> {
            Ok(Box::new(std::io::Cursor::new(self.0.to_vec())))
        }
    }

    impl crate::gopen::Fetch for std::io::Cursor<Vec<u8>> {}

    #[test]
    fn caches_from_a_registered_scheme() {
        // This is the path a subprocess-free build takes: the caller supplies
        // the transport, and the cache works the same way on top of it.
        static SHARD: &[u8] = &{
            let mut body = [0u8; 1024];
            body[257] = b'u';
            body[258] = b's';
            body[259] = b't';
            body[260] = b'a';
            body[261] = b'r';
            body
        };
        crate::gopen::register_scheme("memtest", std::sync::Arc::new(InMemory(SHARD)));

        let dir = tempfile::tempdir().unwrap();
        let cache = FileCache::new(dir.path());

        let first = cache.get_file("memtest://host/shard-000.tar").unwrap();
        assert!(first.exists());
        assert_eq!(std::fs::read(&first).unwrap().len(), 1024);

        assert_eq!(cache.get_file("memtest://host/shard-000.tar").unwrap(), first);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn passes_local_urls_through_untouched() {
        let cache = FileCache::new("/nonexistent-cache");
        assert_eq!(cache.get_file("testdata/a.tar").unwrap(), PathBuf::from("testdata/a.tar"));
    }

    #[cfg(feature = "subprocess")]
    #[test]
    fn rejects_downloads_that_are_not_archives() {
        let cache_dir = tempfile::tempdir().unwrap();
        let cache = FileCache::new(cache_dir.path()).with_retries(1);
        let err = cache.get_file("pipe:printf 'this is not a tar file'").unwrap_err();
        assert!(err.to_string().contains("not an archive"), "{err}");
        assert_eq!(fs::read_dir(cache_dir.path()).unwrap().count(), 0, "the bad download should be removed");
    }

    #[test]
    fn evicts_oldest_files_over_budget() {
        let dir = tempfile::tempdir().unwrap();
        for (i, name) in ["old.bin", "mid.bin", "new.bin"].iter().enumerate() {
            fs::write(dir.path().join(name), vec![b'x'; 1000]).unwrap();
            // Keep creation times distinct and ordered.
            std::thread::sleep(Duration::from_millis(20));
            let _ = i;
        }
        let cleanup = LruCleanup::new(dir.path(), 1500, None);
        let freed = cleanup.cleanup().unwrap();

        assert!(freed >= 1000, "freed {freed}");
        assert!(dir.path().join("new.bin").exists(), "the newest file should survive");
        assert!(!dir.path().join("old.bin").exists(), "the oldest file should be evicted");
    }

    #[test]
    fn throttles_repeated_cleanups() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.bin"), vec![b'x'; 4000]).unwrap();
        let cleanup = LruCleanup::new(dir.path(), 10, Some(Duration::from_secs(60)));
        assert!(cleanup.cleanup().unwrap() > 0);

        fs::write(dir.path().join("b.bin"), vec![b'x'; 4000]).unwrap();
        assert_eq!(cleanup.cleanup().unwrap(), 0, "the second run is inside the interval");
    }
}
