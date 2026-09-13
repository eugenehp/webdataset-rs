//! Transparent decompression of shard streams.
//!
//! Shards are distributed as `.tar`, `.tar.gz`/`.tgz`, and occasionally
//! `.tar.zst`, `.tar.bz2` or `.tar.xz`. Rather than trusting the file
//! extension — which is often absent, as with `pipe:` URLs — the format is
//! detected from the first few bytes, matching the `r|*` mode the Python
//! implementation opens archives with.

use std::io::{Read, Write};

use webdataset_core::error::{Error, Result};
use webdataset_io::cache::{FileType, sniff};

/// How many bytes are needed to identify every supported container.
pub(crate) const SNIFF_LEN: usize = 264;

/// A reader that replays a prefix already consumed from an inner stream.
struct Prefixed<R> {
    prefix: std::io::Cursor<Vec<u8>>,
    rest: R,
}

impl<R: Read> Read for Prefixed<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.prefix.read(buf)?;
        if n > 0 {
            return Ok(n);
        }
        self.rest.read(buf)
    }
}

/// Wrap `stream` in the decoder its magic bytes call for.
///
/// Returns the detected format alongside the decoded stream. An unrecognised
/// stream is passed through unchanged, so an uncompressed tar without a `ustar`
/// magic (old V7 archives) still reads.
pub fn decompress(mut stream: Box<dyn Read + Send>) -> Result<(FileType, Box<dyn Read + Send>)> {
    let mut prefix = vec![0u8; SNIFF_LEN];
    let mut filled = 0;
    while filled < prefix.len() {
        match stream.read(&mut prefix[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    prefix.truncate(filled);

    let kind = sniff(&prefix);
    let combined = Prefixed { prefix: std::io::Cursor::new(prefix), rest: stream };
    let decoded: Box<dyn Read + Send> = match kind {
        // `MultiGzDecoder` keeps reading past the first member, which matters
        // for shards produced by concatenating gzip streams.
        FileType::Gzip => Box::new(flate2::read::MultiGzDecoder::new(combined)),
        #[cfg(feature = "zstd")]
        FileType::Zstd => Box::new(zstd::stream::read::Decoder::new(combined)?),
        #[cfg(feature = "bzip2")]
        FileType::Bzip2 => Box::new(bzip2::read::MultiBzDecoder::new(combined)),
        #[cfg(feature = "xz")]
        FileType::Xz => Box::new(liblzma::read::XzDecoder::new(combined)),
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

/// How a shard being written should be compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Write a plain tar archive.
    #[default]
    None,
    /// gzip, as for `.tar.gz` and `.tgz`.
    Gzip,
    /// Zstandard, as for `.tar.zst`.
    Zstd,
    /// bzip2, as for `.tar.bz2`.
    Bzip2,
    /// XZ, as for `.tar.xz`.
    Xz,
}

impl Compression {
    /// Pick a compression from a file name's extension.
    ///
    /// ```
    /// use webdataset_shard::compress::Compression;
    ///
    /// assert_eq!(Compression::for_name("s-000.tar"), Compression::None);
    /// assert_eq!(Compression::for_name("s-000.tgz"), Compression::Gzip);
    /// assert_eq!(Compression::for_name("s-000.tar.gz"), Compression::Gzip);
    /// ```
    pub fn for_name(name: &str) -> Compression {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".tgz") || lower.ends_with(".gz") {
            Compression::Gzip
        } else if lower.ends_with(".zst") || lower.ends_with(".zstd") {
            Compression::Zstd
        } else if lower.ends_with(".bz2") || lower.ends_with(".tbz") {
            Compression::Bzip2
        } else if lower.ends_with(".xz") {
            Compression::Xz
        } else {
            Compression::None
        }
    }

    /// Wrap `out` in the matching encoder.
    pub fn wrap(self, out: Box<dyn Write + Send>) -> Result<Box<dyn Finish>> {
        Ok(match self {
            Compression::None => Box::new(Plain(out)),
            Compression::Gzip => Box::new(flate2::write::GzEncoder::new(out, flate2::Compression::default())),
            #[cfg(feature = "zstd")]
            Compression::Zstd => Box::new(zstd::stream::write::Encoder::new(out, 0)?.auto_finish()),
            #[cfg(feature = "bzip2")]
            Compression::Bzip2 => Box::new(bzip2::write::BzEncoder::new(out, bzip2::Compression::default())),
            #[cfg(feature = "xz")]
            Compression::Xz => Box::new(liblzma::write::XzEncoder::new(out, 6)),
            #[allow(unreachable_patterns)]
            other => {
                return Err(Error::unsupported(format!(
                    "writing {other:?} shards; rebuild webdataset-shard with the matching feature enabled"
                )));
            }
        })
    }
}

/// A writer that must be told when the stream is complete.
pub trait Finish: Write + Send {
    /// Flush any trailing compressed data.
    fn finish_stream(self: Box<Self>) -> Result<()>;
}

/// An uncompressed passthrough writer.
struct Plain(Box<dyn Write + Send>);

impl Write for Plain {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush()
    }
}

impl Finish for Plain {
    fn finish_stream(mut self: Box<Self>) -> Result<()> {
        self.0.flush()?;
        Ok(())
    }
}

impl<W: Write + Send> Finish for flate2::write::GzEncoder<W> {
    fn finish_stream(self: Box<Self>) -> Result<()> {
        let mut inner = (*self).finish()?;
        inner.flush()?;
        Ok(())
    }
}

#[cfg(feature = "zstd")]
impl<W: Write + Send> Finish for zstd::stream::AutoFinishEncoder<'static, W> {
    fn finish_stream(mut self: Box<Self>) -> Result<()> {
        self.flush()?;
        Ok(())
    }
}

#[cfg(feature = "bzip2")]
impl<W: Write + Send> Finish for bzip2::write::BzEncoder<W> {
    fn finish_stream(self: Box<Self>) -> Result<()> {
        let mut inner = (*self).finish()?;
        inner.flush()?;
        Ok(())
    }
}

#[cfg(feature = "xz")]
impl<W: Write + Send> Finish for liblzma::write::XzEncoder<W> {
    fn finish_stream(self: Box<Self>) -> Result<()> {
        let mut inner = (*self).finish()?;
        inner.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_gzip_and_decodes_it() {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(b"hello gzip").unwrap();
        let compressed = encoder.finish().unwrap();

        let (kind, mut stream) = decompress(Box::new(std::io::Cursor::new(compressed))).unwrap();
        assert_eq!(kind, FileType::Gzip);
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        assert_eq!(out, "hello gzip");
    }

    #[test]
    fn passes_plain_streams_through() {
        let (kind, mut stream) = decompress(Box::new(std::io::Cursor::new(b"plain bytes".to_vec()))).unwrap();
        assert_eq!(kind, FileType::Unknown);
        let mut out = String::new();
        stream.read_to_string(&mut out).unwrap();
        assert_eq!(out, "plain bytes");
    }

    #[test]
    fn handles_streams_shorter_than_the_sniff_window() {
        let (_, mut stream) = decompress(Box::new(std::io::Cursor::new(b"ab".to_vec()))).unwrap();
        let mut out = Vec::new();
        stream.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"ab");
    }

    /// A `Vec<u8>` that a boxed `'static` writer can hand back to the test.
    #[derive(Clone, Default)]
    struct SharedBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buffer lock").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Write with `compression`, read it back, and check both directions agree.
    #[allow(dead_code)]
    fn round_trip(compression: Compression, expected: FileType) {
        let buffer = SharedBuffer::default();
        let mut out = compression.wrap(Box::new(buffer.clone())).unwrap();
        out.write_all(b"payload that compresses").unwrap();
        out.finish_stream().unwrap();

        let bytes = buffer.0.lock().expect("buffer lock").clone();
        let (kind, mut back) = decompress(Box::new(std::io::Cursor::new(bytes))).unwrap();
        assert_eq!(kind, expected);

        let mut text = String::new();
        back.read_to_string(&mut text).unwrap();
        assert_eq!(text, "payload that compresses");
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn round_trips_through_zstd() {
        round_trip(Compression::Zstd, FileType::Zstd);
    }

    #[cfg(feature = "bzip2")]
    #[test]
    fn round_trips_through_bzip2() {
        round_trip(Compression::Bzip2, FileType::Bzip2);
    }

    #[cfg(feature = "xz")]
    #[test]
    fn round_trips_through_xz() {
        round_trip(Compression::Xz, FileType::Xz);
    }

    #[test]
    fn names_map_to_compressions() {
        assert_eq!(Compression::for_name("a.tar.zst"), Compression::Zstd);
        assert_eq!(Compression::for_name("a.tar.bz2"), Compression::Bzip2);
        assert_eq!(Compression::for_name("a.tar.xz"), Compression::Xz);
        assert_eq!(Compression::for_name("A.TGZ"), Compression::Gzip);
    }

    #[test]
    fn round_trips_through_the_gzip_writer() {
        let buffer = SharedBuffer::default();
        let mut out = Compression::Gzip.wrap(Box::new(buffer.clone())).unwrap();
        out.write_all(b"round trip").unwrap();
        out.finish_stream().unwrap();

        let bytes = buffer.0.lock().expect("buffer lock").clone();
        let (kind, mut back) = decompress(Box::new(std::io::Cursor::new(bytes))).unwrap();
        assert_eq!(kind, FileType::Gzip);
        let mut text = String::new();
        back.read_to_string(&mut text).unwrap();
        assert_eq!(text, "round trip");
    }
}
