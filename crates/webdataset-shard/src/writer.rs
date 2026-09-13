//! Writing samples into tar archives.
//!
//! [`TarWriter`] turns samples back into tar members, naming each one
//! `<key>.<extension>`; [`ShardWriter`] rolls over to a new archive once a
//! shard reaches a size or count limit.
//!
//! ```
//! use webdataset_core::{Sample, Value};
//! use webdataset_shard::{ShardWriter, TarWriter};
//!
//! let dir = tempfile::tempdir()?;
//! let pattern = dir.path().join("train-%06d.tar");
//!
//! let mut writer = ShardWriter::new(pattern.to_str().unwrap())?.with_max_count(1000);
//! let mut sample = Sample::with_key("sample0001");
//! sample.insert("txt", Value::Bytes("a caption".into()));
//! writer.write(&sample)?;
//! writer.close()?;
//!
//! assert!(dir.path().join("train-000000.tar").exists());
//! # Ok::<(), webdataset_core::Error>(())
//! ```

use std::io::Write;
use std::sync::Arc;

use bytes::Bytes;
use webdataset_core::error::{Error, Result};
use webdataset_core::sample::{KEY, Sample};
use webdataset_core::utils::format_shard_pattern;
use webdataset_core::value::Value;
use webdataset_io::gopen::gopen_write;

use crate::compress::{Compression, Finish};

/// Turns a sample field into the bytes stored in the archive.
///
/// The default [`RawEncoder`] only accepts values that are already bytes or
/// text; the `webdataset` crate supplies a full encoder that understands
/// `.json`, `.npy`, `.ten`, images, and the rest.
pub trait Encoder: Send + Sync {
    /// Encode `value`, which was stored under the given file `extension`.
    fn encode(&self, extension: &str, value: &Value) -> Result<Bytes>;
}

/// Stores bytes and text verbatim and rejects everything else.
#[derive(Debug, Clone, Copy, Default)]
pub struct RawEncoder;

impl Encoder for RawEncoder {
    fn encode(&self, extension: &str, value: &Value) -> Result<Bytes> {
        match value {
            Value::Bytes(b) => Ok(b.clone()),
            Value::Text(s) => Ok(Bytes::from(s.clone().into_bytes())),
            other => Err(Error::encode(
                extension,
                format!("the raw encoder only stores bytes and text, not {}", other.type_name()),
            )),
        }
    }
}

/// Writes samples into a single tar archive.
pub struct TarWriter {
    builder: Option<tar::Builder<Box<dyn Finish>>>,
    encoder: Arc<dyn Encoder>,
    user: String,
    group: String,
    mode: u32,
    mtime: Option<u64>,
    keep_meta: bool,
}

impl std::fmt::Debug for TarWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TarWriter")
            .field("user", &self.user)
            .field("group", &self.group)
            .field("mode", &format_args!("{:o}", self.mode))
            .field("mtime", &self.mtime)
            .finish_non_exhaustive()
    }
}

impl TarWriter {
    /// Write to `out`, applying `compression`.
    pub fn new(out: Box<dyn Write + Send>, compression: Compression) -> Result<TarWriter> {
        Ok(TarWriter {
            builder: Some(tar::Builder::new(compression.wrap(out)?)),
            encoder: Arc::new(RawEncoder),
            user: "bigdata".into(),
            group: "bigdata".into(),
            mode: 0o444,
            mtime: None,
            keep_meta: false,
        })
    }

    /// Create the archive at `url`, choosing compression from its name.
    ///
    /// Any URL [`gopen_write`] understands works, so shards can be written
    /// straight to cloud storage with `pipe:` or `gs:`.
    pub fn create(url: &str) -> Result<TarWriter> {
        let compression = Compression::for_name(url);
        TarWriter::new(Box::new(SinkWriter(gopen_write(url)?)), compression)
    }

    /// Encode values with `encoder` instead of [`RawEncoder`].
    pub fn with_encoder(mut self, encoder: Arc<dyn Encoder>) -> TarWriter {
        self.encoder = encoder;
        self
    }

    /// Set the owner recorded in each tar header.
    pub fn with_owner(mut self, user: &str, group: &str) -> TarWriter {
        self.user = user.to_string();
        self.group = group.to_string();
        self
    }

    /// Set the permission bits recorded in each tar header.
    pub fn with_mode(mut self, mode: u32) -> TarWriter {
        self.mode = mode;
        self
    }

    /// Pin the modification time, which makes the output byte-reproducible.
    pub fn with_mtime(mut self, mtime: Option<u64>) -> TarWriter {
        self.mtime = mtime;
        self
    }

    /// Also write metadata fields, whose names begin with `_`.
    pub fn keep_meta(mut self, keep: bool) -> TarWriter {
        self.keep_meta = keep;
        self
    }

    /// Append a sample, returning the number of payload bytes written.
    ///
    /// Fields are written in sorted order so that archives built from the same
    /// samples are identical.
    pub fn write(&mut self, sample: &Sample) -> Result<u64> {
        let key = sample
            .key()
            .ok_or_else(|| Error::value("a sample must have a __key__ before it can be written"))?
            .to_string();

        let mut names: Vec<&str> =
            sample.keys().filter(|name| *name != KEY).filter(|name| self.keep_meta || !name.starts_with('_')).collect();
        names.sort_unstable();

        let now = self.mtime.unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
        });

        let mut total = 0u64;
        for name in names {
            let value = sample.get(name).expect("name came from the sample");
            let data = self.encoder.encode(name, value)?;

            let mut header = tar::Header::new_ustar();
            header.set_size(data.len() as u64);
            header.set_mode(self.mode);
            header.set_mtime(now);
            header.set_uid(0);
            header.set_gid(0);
            header.set_entry_type(tar::EntryType::Regular);
            header.set_username(&self.user).map_err(|e| Error::encode(name, e))?;
            header.set_groupname(&self.group).map_err(|e| Error::encode(name, e))?;

            let path = format!("{key}.{name}");
            let builder = self.builder.as_mut().ok_or_else(|| Error::value("writer is already closed"))?;
            builder.append_data(&mut header, &path, &data[..]).map_err(|e| Error::encode(name, e))?;
            total += data.len() as u64;
        }
        Ok(total)
    }

    /// Finish the archive: write the trailer and flush any compression.
    pub fn close(mut self) -> Result<()> {
        self.close_inner()
    }

    fn close_inner(&mut self) -> Result<()> {
        let Some(builder) = self.builder.take() else {
            return Ok(());
        };
        let inner = builder.into_inner().map_err(|e| Error::Io(e).context("finishing the archive"))?;
        inner.finish_stream()
    }
}

impl Drop for TarWriter {
    fn drop(&mut self) {
        if let Err(e) = self.close_inner() {
            log::error!("failed to finish archive: {e}");
        }
    }
}

/// Adapts a [`Sink`](webdataset_io::gopen::Sink) into a plain writer.
struct SinkWriter(Box<dyn webdataset_io::gopen::Sink>);

impl Write for SinkWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl Drop for SinkWriter {
    fn drop(&mut self) {
        if let Err(e) = self.0.finish() {
            log::error!("failed to close output stream: {e}");
        }
    }
}

/// Called after each shard is finished.
pub type PostHook = Box<dyn FnMut(&str) -> Result<()> + Send>;

/// Writes samples across a numbered series of shards.
///
/// A new shard is started once the current one reaches `max_count` samples or
/// `max_size` payload bytes, so the resulting shards stay in the size range
/// that streams efficiently (a few hundred megabytes is typical).
pub struct ShardWriter {
    pattern: String,
    max_count: usize,
    max_size: u64,
    shard: usize,
    count: usize,
    size: u64,
    total: usize,
    current: Option<TarWriter>,
    name: Option<String>,
    encoder: Arc<dyn Encoder>,
    post: Option<PostHook>,
}

impl std::fmt::Debug for ShardWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardWriter")
            .field("pattern", &self.pattern)
            .field("max_count", &self.max_count)
            .field("max_size", &self.max_size)
            .field("shard", &self.shard)
            .field("total", &self.total)
            .finish_non_exhaustive()
    }
}

impl ShardWriter {
    /// Write shards named by a `printf`-style pattern such as `out-%06d.tar`.
    pub fn new(pattern: &str) -> Result<ShardWriter> {
        // Fail now rather than after the caller has streamed a million samples.
        format_shard_pattern(pattern, 0)?;
        Ok(ShardWriter {
            pattern: pattern.to_string(),
            max_count: 100_000,
            max_size: 3_000_000_000,
            shard: 0,
            count: 0,
            size: 0,
            total: 0,
            current: None,
            name: None,
            encoder: Arc::new(RawEncoder),
            post: None,
        })
    }

    /// Start numbering at `shard` instead of zero.
    pub fn with_start_shard(mut self, shard: usize) -> ShardWriter {
        self.shard = shard;
        self
    }

    /// Roll over after this many samples.
    pub fn with_max_count(mut self, max_count: usize) -> ShardWriter {
        self.max_count = max_count;
        self
    }

    /// Roll over after this many payload bytes.
    pub fn with_max_size(mut self, max_size: u64) -> ShardWriter {
        self.max_size = max_size;
        self
    }

    /// Encode values with `encoder` instead of [`RawEncoder`].
    pub fn with_encoder(mut self, encoder: Arc<dyn Encoder>) -> ShardWriter {
        self.encoder = encoder;
        self
    }

    /// Run `post` on each shard's name once that shard is complete.
    ///
    /// Useful for uploading or verifying shards as they are produced.
    pub fn with_post_hook(mut self, post: PostHook) -> ShardWriter {
        self.post = Some(post);
        self
    }

    /// The name of the shard currently being written.
    pub fn current_shard(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// How many samples have been written across all shards.
    pub fn total(&self) -> usize {
        self.total
    }

    /// How many shards have been started.
    pub fn shard_count(&self) -> usize {
        self.shard
    }

    /// Append a sample, rolling over to a new shard if the limits are reached.
    pub fn write(&mut self, sample: &Sample) -> Result<()> {
        if self.current.is_none() || self.count >= self.max_count || self.size >= self.max_size {
            self.next_shard()?;
        }
        let writer = self.current.as_mut().expect("next_shard installed a writer");
        let size = writer.write(sample)?;
        self.count += 1;
        self.total += 1;
        self.size += size;
        Ok(())
    }

    /// Finish the current shard and open the next one.
    pub fn next_shard(&mut self) -> Result<()> {
        self.finish()?;
        let name = format_shard_pattern(&self.pattern, self.shard)?;
        log::debug!("writing shard {name}");
        self.shard += 1;
        self.current = Some(TarWriter::create(&name)?.with_encoder(self.encoder.clone()));
        self.name = Some(name);
        self.count = 0;
        self.size = 0;
        Ok(())
    }

    /// Finish the current shard, if one is open.
    fn finish(&mut self) -> Result<()> {
        let Some(writer) = self.current.take() else {
            return Ok(());
        };
        writer.close()?;
        if let (Some(post), Some(name)) = (self.post.as_mut(), self.name.as_ref()) {
            post(name)?;
        }
        Ok(())
    }

    /// Finish the last shard and release the writer.
    pub fn close(mut self) -> Result<()> {
        self.finish()
    }
}

impl Drop for ShardWriter {
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            log::error!("failed to finish shard: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::{TarFiles, group_by_keys};

    fn sample(key: &str, text: &str) -> Sample {
        let mut s = Sample::with_key(key);
        s.insert("txt", Value::Bytes(Bytes::from(text.to_string())));
        s.insert("cls", Value::Bytes(Bytes::from_static(b"7")));
        s
    }

    fn read_back(path: &std::path::Path) -> Vec<Sample> {
        group_by_keys(TarFiles::open(path.to_str().unwrap()).unwrap()).map(|s| s.unwrap()).collect()
    }

    #[test]
    fn round_trips_samples_through_a_tar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tar");

        let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap();
        writer.write(&sample("a", "first")).unwrap();
        writer.write(&sample("b", "second")).unwrap();
        writer.close().unwrap();

        let back = read_back(&path);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].key(), Some("a"));
        assert_eq!(back[0].get("txt").unwrap().as_bytes().unwrap().as_ref(), b"first");
        assert_eq!(back[1].get("cls").unwrap().as_bytes().unwrap().as_ref(), b"7");
    }

    #[test]
    fn round_trips_through_a_compressed_tar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tar.gz");

        let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap();
        writer.write(&sample("a", "compressed")).unwrap();
        writer.close().unwrap();

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(&raw[..2], &[0x1f, 0x8b], "the shard should be gzip compressed");
        assert_eq!(read_back(&path)[0].get("txt").unwrap().as_bytes().unwrap().as_ref(), b"compressed");
    }

    #[test]
    fn writes_fields_in_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tar");

        let mut s = Sample::with_key("k");
        s.insert("zzz", Value::Bytes(Bytes::from_static(b"z")));
        s.insert("aaa", Value::Bytes(Bytes::from_static(b"a")));

        let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap();
        writer.write(&s).unwrap();
        writer.close().unwrap();

        let names: Vec<String> = TarFiles::open(path.to_str().unwrap()).unwrap().map(|f| f.unwrap().name).collect();
        assert_eq!(names, ["k.aaa", "k.zzz"]);
    }

    #[test]
    fn skips_metadata_unless_asked_to_keep_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tar");

        let mut s = Sample::with_key("k");
        s.insert("txt", Value::Bytes(Bytes::from_static(b"x")));
        s.set_url("http://host/shard.tar");

        let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap();
        writer.write(&s).unwrap();
        writer.close().unwrap();

        let names: Vec<String> = TarFiles::open(path.to_str().unwrap()).unwrap().map(|f| f.unwrap().name).collect();
        assert_eq!(names, ["k.txt"], "__url__ is metadata and must not be stored");
    }

    #[test]
    fn refuses_samples_without_a_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.tar");
        let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap();

        let mut s = Sample::new();
        s.insert("txt", Value::Bytes(Bytes::from_static(b"x")));
        assert!(writer.write(&s).is_err());
    }

    #[test]
    fn produces_identical_output_for_a_pinned_mtime() {
        let dir = tempfile::tempdir().unwrap();
        let mut written = Vec::new();
        for name in ["a.tar", "b.tar"] {
            let path = dir.path().join(name);
            let mut writer = TarWriter::create(path.to_str().unwrap()).unwrap().with_mtime(Some(0));
            writer.write(&sample("k", "same")).unwrap();
            writer.close().unwrap();
            written.push(std::fs::read(&path).unwrap());
        }
        assert_eq!(written[0], written[1], "a pinned mtime should make the output reproducible");
    }

    #[test]
    fn rolls_over_at_the_sample_limit() {
        let dir = tempfile::tempdir().unwrap();
        let pattern = dir.path().join("shard-%04d.tar");

        let mut writer = ShardWriter::new(pattern.to_str().unwrap()).unwrap().with_max_count(2);
        for i in 0..5 {
            writer.write(&sample(&format!("k{i}"), "x")).unwrap();
        }
        assert_eq!(writer.total(), 5);
        writer.close().unwrap();

        let counts: Vec<usize> = ["shard-0000.tar", "shard-0001.tar", "shard-0002.tar"]
            .iter()
            .map(|name| read_back(&dir.path().join(name)).len())
            .collect();
        assert_eq!(counts, [2, 2, 1]);
        assert!(!dir.path().join("shard-0003.tar").exists());
    }

    #[test]
    fn rolls_over_at_the_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let pattern = dir.path().join("shard-%d.tar");

        let mut writer = ShardWriter::new(pattern.to_str().unwrap()).unwrap().with_max_size(10);
        for i in 0..4 {
            writer.write(&sample(&format!("k{i}"), "0123456789")).unwrap();
        }
        writer.close().unwrap();
        assert!(dir.path().join("shard-3.tar").exists(), "each sample exceeds the size limit on its own");
    }

    #[test]
    fn runs_the_post_hook_for_every_finished_shard() {
        let dir = tempfile::tempdir().unwrap();
        let pattern = dir.path().join("shard-%d.tar");
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));

        let recorder = seen.clone();
        let mut writer = ShardWriter::new(pattern.to_str().unwrap()).unwrap().with_max_count(1).with_post_hook(
            Box::new(move |name| {
                recorder.lock().expect("lock").push(name.to_string());
                Ok(())
            }),
        );
        writer.write(&sample("a", "x")).unwrap();
        writer.write(&sample("b", "x")).unwrap();
        writer.close().unwrap();

        assert_eq!(seen.lock().expect("lock").len(), 2);
    }

    #[test]
    fn rejects_patterns_without_a_conversion() {
        assert!(ShardWriter::new("no-number.tar").is_err());
    }

    #[test]
    fn raw_encoder_refuses_structured_values() {
        assert!(RawEncoder.encode("json", &Value::Int(3)).is_err());
        assert!(RawEncoder.encode("txt", &Value::Text("ok".into())).is_ok());
    }
}
