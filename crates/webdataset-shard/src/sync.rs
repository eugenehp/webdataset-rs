//! A synchronous, streaming tar parser.
//!
//! The archive is read forward, one 512-byte block at a time, and never seeked.
//! That is what lets a shard be read from a pipe or an HTTP response, and it is
//! also what makes the reader usable on targets without threads — WebAssembly
//! in particular, where the read-ahead thread used elsewhere is unavailable.
//!
//! The formats a WebDataset shard can arrive in are all handled: plain ustar,
//! GNU long names (`L` records), and PAX extended headers (`x` records).
//! Directories, symlinks, and device nodes are skipped, since a sample is made
//! of regular files.

use std::io::Read;

use webdataset_core::error::{Error, Result};

use crate::prelude::*;

/// The size of a tar header, and the unit every entry is padded to.
pub(crate) const BLOCK: usize = 512;

/// The most to allocate before reading anything, so that a header claiming an
/// absurd size costs nothing until the bytes to back it actually turn up.
pub(crate) const MAX_PREALLOC: usize = 1 << 20;

/// How much to add to the buffer at a time beyond that point.
///
/// Matched to [`MAX_PREALLOC`] so that a genuine multi-megabyte entry costs a
/// handful of iterations rather than hundreds, while the memory an entry can
/// claim before producing any data stays bounded by the two together.
pub(crate) const CHUNK: usize = 1 << 20;

/// The offset of the type flag, which says what kind of entry follows.
#[cfg(feature = "async")]
pub(crate) const TYPEFLAG: usize = field::TYPEFLAG;

/// Offsets into the header block.
mod field {
    pub const NAME: core::ops::Range<usize> = 0..100;
    pub const SIZE: core::ops::Range<usize> = 124..136;
    pub const CHECKSUM: core::ops::Range<usize> = 148..156;
    pub const TYPEFLAG: usize = 156;
    pub const MAGIC: core::ops::Range<usize> = 257..263;
    pub const PREFIX: core::ops::Range<usize> = 345..500;
}

/// One regular file, as it appears in the archive.
#[derive(Debug, Clone)]
pub struct RawEntry {
    /// The member's full path.
    pub name: String,
    /// The member's contents.
    pub data: Vec<u8>,
}

/// The regular files of a tar archive, read in order.
pub struct TarEntries<R> {
    reader: R,
    done: bool,
    /// A name supplied by a preceding GNU `L` record.
    long_name: Option<String>,
    /// A path supplied by a preceding PAX `x` record.
    pax_path: Option<String>,
}

impl<R: Read> TarEntries<R> {
    /// Parse `reader` as an uncompressed tar archive.
    pub fn new(reader: R) -> TarEntries<R> {
        TarEntries { reader, done: false, long_name: None, pax_path: None }
    }

    /// Read the next regular file, or `None` at the end of the archive.
    fn advance(&mut self) -> Result<Option<RawEntry>> {
        loop {
            let Some(header) = self.read_header()? else {
                return Ok(None);
            };

            let size = parse_size(&header)?;
            let typeflag = header[field::TYPEFLAG];

            match typeflag {
                // GNU long name: the payload names the entry that follows.
                b'L' => {
                    self.long_name = Some(trim_nul(&self.read_payload(size)?)?);
                    continue;
                }
                // GNU long link name: irrelevant to us, but must be consumed.
                b'K' => {
                    self.read_payload(size)?;
                    continue;
                }
                // PAX extended header: may override the following entry's path.
                b'x' | b'X' => {
                    let records = self.read_payload(size)?;
                    if let Some(path) = pax_path(&records) {
                        self.pax_path = Some(path);
                    }
                    continue;
                }
                // PAX global header: applies to the whole archive; skipped.
                b'g' => {
                    self.read_payload(size)?;
                    continue;
                }
                // Regular files, in both the modern and the historic spelling.
                b'0' | b'\0' | b'7' => {}
                // Directories, links, devices and fifos hold no sample data.
                _ => {
                    self.read_payload(size)?;
                    self.long_name = None;
                    self.pax_path = None;
                    continue;
                }
            }

            let name = match self.pax_path.take().or_else(|| self.long_name.take()) {
                Some(name) => name,
                None => header_name(&header)?,
            };
            return Ok(Some(RawEntry { name, data: self.read_payload(size)? }));
        }
    }

    /// Read one header block, returning `None` at the end-of-archive marker.
    fn read_header(&mut self) -> Result<Option<[u8; BLOCK]>> {
        let mut header = [0u8; BLOCK];
        match self.fill(&mut header)? {
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
    fn read_payload(&mut self, size: u64) -> Result<Vec<u8>> {
        let size = usize::try_from(size).map_err(|_| Error::format("tar entry is too large for this platform"))?;
        // Grow with the bytes that actually arrive rather than reserving what
        // the header claims. A shard is untrusted input, and a crafted header
        // can name a size no machine can allocate; reserving that up front
        // aborts the process, which no error handler can catch. Growing as the
        // data arrives turns the same input into an ordinary error.
        let mut data = Vec::with_capacity(size.min(MAX_PREALLOC));
        while data.len() < size {
            let start = data.len();
            data.resize(start + CHUNK.min(size - start), 0);
            if !matches!(self.fill(&mut data[start..])?, Filled::Complete) {
                return Err(Error::format(format!("tar archive ends mid-entry, expected {size} bytes")));
            }
        }
        let padding = (BLOCK - size % BLOCK) % BLOCK;
        if padding > 0 {
            let mut skip = [0u8; BLOCK];
            if !matches!(self.fill(&mut skip[..padding])?, Filled::Complete) {
                return Err(Error::format("tar archive ends mid-padding"));
            }
        }
        Ok(data)
    }

    /// Read exactly `buf.len()` bytes, distinguishing a clean end of stream.
    fn fill(&mut self, buf: &mut [u8]) -> Result<Filled> {
        let mut filled = 0;
        while filled < buf.len() {
            match self.reader.read(&mut buf[filled..])? {
                0 if filled == 0 => return Ok(Filled::Eof),
                0 => return Ok(Filled::Partial(filled)),
                n => filled += n,
            }
        }
        Ok(Filled::Complete)
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

impl<R: Read> Iterator for TarEntries<R> {
    type Item = Result<RawEntry>;

    fn next(&mut self) -> Option<Result<RawEntry>> {
        if self.done {
            return None;
        }
        match self.advance() {
            Ok(Some(entry)) => Some(Ok(entry)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                // A tar reader cannot resynchronise mid-stream, so one bad
                // header ends the archive.
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// The header's own checksum, which is how a non-tar stream is detected.
///
/// The field is computed with the checksum bytes themselves treated as spaces.
/// Historic implementations disagreed on whether the bytes are signed, so both
/// readings are accepted.
pub(crate) fn verify_checksum(header: &[u8; BLOCK]) -> Result<()> {
    let stored =
        parse_octal(&header[field::CHECKSUM]).ok_or_else(|| Error::format("tar header has an unreadable checksum"))?;

    let mut unsigned: u64 = 0;
    let mut signed: i64 = 0;
    for (i, byte) in header.iter().enumerate() {
        let value = if field::CHECKSUM.contains(&i) { b' ' } else { *byte };
        unsigned += value as u64;
        signed += value as i8 as i64;
    }

    if stored == unsigned || stored as i64 == signed {
        return Ok(());
    }
    Err(Error::format(format!("tar header checksum mismatch: stored {stored}, computed {unsigned}")))
}

/// The entry's name, joining the ustar `prefix` field when one is present.
pub(crate) fn header_name(header: &[u8; BLOCK]) -> Result<String> {
    let name = trim_nul(&header[field::NAME])?;
    let ustar = header[field::MAGIC].starts_with(b"ustar");
    if !ustar {
        return Ok(name);
    }
    let prefix = trim_nul(&header[field::PREFIX])?;
    if prefix.is_empty() { Ok(name) } else { Ok(format!("{prefix}/{name}")) }
}

/// The entry's payload size, in either octal or GNU base-256 form.
pub(crate) fn parse_size(header: &[u8; BLOCK]) -> Result<u64> {
    let raw = &header[field::SIZE];
    // GNU marks binary sizes by setting the top bit of the first byte.
    if raw[0] & 0x80 != 0 {
        let mut size: u64 = 0;
        for byte in &raw[raw.len() - 8..] {
            size = (size << 8) | *byte as u64;
        }
        return Ok(size);
    }
    parse_octal(raw).ok_or_else(|| Error::format("tar header has an unreadable size"))
}

/// Read a NUL- or space-padded octal field.
fn parse_octal(field: &[u8]) -> Option<u64> {
    let text = field.iter().copied().take_while(|b| *b != 0).collect::<Vec<u8>>();
    let text = core::str::from_utf8(&text).ok()?.trim();
    if text.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(text, 8).ok()
}

/// Read a NUL-terminated string field.
pub(crate) fn trim_nul(field: &[u8]) -> Result<String> {
    let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
    Ok(core::str::from_utf8(&field[..end])?.to_string())
}

/// The `path=` record of a PAX extended header, if it has one.
///
/// Records are `"<length> <key>=<value>\n"`, with `<length>` counting the whole
/// record including itself.
pub(crate) fn pax_path(records: &[u8]) -> Option<String> {
    let mut rest = records;
    while !rest.is_empty() {
        let space = rest.iter().position(|b| *b == b' ')?;
        let length: usize = core::str::from_utf8(&rest[..space]).ok()?.parse().ok()?;
        if length == 0 || length > rest.len() {
            return None;
        }
        let record = &rest[space + 1..length];
        let record = record.strip_suffix(b"\n").unwrap_or(record);
        if let Some(value) = record.strip_prefix(b"path=") {
            return core::str::from_utf8(value).ok().map(str::to_string);
        }
        rest = &rest[length..];
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build an archive with the `tar` crate, so the parser is checked against
    /// a independent implementation of the format.
    fn build(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut builder = tar::Builder::new(Vec::new());
        for (name, data) in entries {
            let mut header = tar::Header::new_ustar();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Regular);
            builder.append_data(&mut header, name, *data).unwrap();
        }
        builder.into_inner().unwrap()
    }

    fn parse(archive: &[u8]) -> Vec<RawEntry> {
        TarEntries::new(std::io::Cursor::new(archive.to_vec())).map(|e| e.unwrap()).collect()
    }

    #[test]
    fn reads_the_entries_the_tar_crate_wrote() {
        let archive = build(&[("a.txt", b"first"), ("b.bin", b"second"), ("c.json", b"{}")]);
        let entries = parse(&archive);

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].name, "a.txt");
        assert_eq!(entries[0].data, b"first");
        assert_eq!(entries[2].name, "c.json");
        assert_eq!(entries[2].data, b"{}");
    }

    #[test]
    fn handles_sizes_around_the_block_boundary() {
        for size in [0usize, 1, 511, 512, 513, 1024, 4096] {
            let data = vec![b'x'; size];
            let archive = build(&[("a.bin", &data), ("b.bin", b"after")]);
            let entries = parse(&archive);
            assert_eq!(entries.len(), 2, "size {size}");
            assert_eq!(entries[0].data.len(), size, "size {size}");
            assert_eq!(entries[1].data, b"after", "size {size}: padding was miscounted");
        }
    }

    #[test]
    fn reads_names_stored_in_the_ustar_prefix() {
        let deep = format!("{}/leaf.txt", "directory".repeat(12));
        assert!(deep.len() > 100, "the name must not fit in the name field");
        let archive = build(&[(deep.as_str(), b"deep")]);

        let entries = parse(&archive);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, deep);
    }

    #[test]
    fn reads_gnu_long_names() {
        // A name too long even for the prefix field forces a GNU `L` record.
        let long = format!("{}.txt", "n".repeat(300));
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(4);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_entry_type(tar::EntryType::Regular);
        builder.append_data(&mut header, &long, &b"data"[..]).unwrap();
        let archive = builder.into_inner().unwrap();

        let entries = parse(&archive);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, long);
        assert_eq!(entries[0].data, b"data");
    }

    #[test]
    fn skips_directories_and_keeps_the_files_after_them() {
        let mut builder = tar::Builder::new(Vec::new());
        let mut dir = tar::Header::new_ustar();
        dir.set_size(0);
        dir.set_mtime(0);
        dir.set_entry_type(tar::EntryType::Directory);
        builder.append_data(&mut dir, "adir/", &[][..]).unwrap();

        let mut file = tar::Header::new_ustar();
        file.set_size(5);
        file.set_mtime(0);
        file.set_entry_type(tar::EntryType::Regular);
        builder.append_data(&mut file, "adir/f.txt", &b"hello"[..]).unwrap();
        let archive = builder.into_inner().unwrap();

        let entries = parse(&archive);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "adir/f.txt");
    }

    #[test]
    fn stops_cleanly_at_the_end_of_archive_marker() {
        let archive = build(&[("a.txt", b"x")]);
        assert!(archive.len() > 512 * 3, "the trailer should be present");
        assert_eq!(parse(&archive).len(), 1);
    }

    #[test]
    fn rejects_a_stream_that_is_not_a_tar() {
        let junk = vec![b'x'; 4096];
        let outcome: Vec<_> = TarEntries::new(std::io::Cursor::new(junk)).collect();
        assert_eq!(outcome.len(), 1);
        assert!(outcome[0].is_err(), "a bad checksum should be reported");
    }

    #[test]
    fn reports_a_truncated_archive() {
        let mut archive = build(&[("a.txt", b"0123456789")]);
        archive.truncate(512 + 4);
        let outcome: Vec<_> = TarEntries::new(std::io::Cursor::new(archive)).collect();
        assert!(outcome.last().unwrap().is_err(), "a truncated entry should be reported");
    }

    #[test]
    fn reads_an_empty_archive() {
        let archive = build(&[]);
        assert_eq!(parse(&archive).len(), 0);
        assert_eq!(parse(&[]).len(), 0);
    }

    /// Build a PAX record, whose length prefix counts the whole record.
    fn pax_record(body: &str) -> String {
        let mut length = body.len() + 2; // the body, a space, and the newline
        loop {
            let record = format!("{length} {body}\n");
            if record.len() == length {
                return record;
            }
            length = record.len();
        }
    }

    #[test]
    fn parses_pax_path_records() {
        let records = format!("{}{}", pax_record("comment=ignored"), pax_record("path=some/long/name.txt"));
        assert_eq!(pax_path(records.as_bytes()).as_deref(), Some("some/long/name.txt"));

        assert_eq!(pax_path(pax_record("comment=nothing").as_bytes()), None);
        assert_eq!(pax_path(b""), None);
        assert_eq!(pax_path(b"garbage without a length"), None);
    }

    #[test]
    fn reads_a_pax_named_entry() {
        // The `tar` crate emits a PAX header for names with non-ASCII bytes.
        let name = "sämple/ünïcode-name.txt";
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(3);
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_entry_type(tar::EntryType::Regular);
        builder.append_data(&mut header, name, &b"abc"[..]).unwrap();
        let archive = builder.into_inner().unwrap();

        let entries = parse(&archive);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, name);
        assert_eq!(entries[0].data, b"abc");
    }

    #[test]
    fn parses_octal_and_base256_sizes() {
        let mut header = [0u8; BLOCK];
        header[field::SIZE.start..field::SIZE.start + 12].copy_from_slice(b"00000000144\0");
        assert_eq!(parse_size(&header).unwrap(), 0o144);

        let mut header = [0u8; BLOCK];
        header[field::SIZE.start] = 0x80;
        header[field::SIZE.end - 1] = 9;
        assert_eq!(parse_size(&header).unwrap(), 9);
    }

    #[test]
    fn writes_and_reads_a_realistic_shard() {
        // 200 entries with mixed sizes, then read them all back.
        let bodies: Vec<Vec<u8>> = (0..200).map(|i| vec![(i % 251) as u8; i * 7 % 3000]).collect();
        let names: Vec<String> = (0..200).map(|i| format!("sample{i:04}.bin")).collect();
        let entries: Vec<(&str, &[u8])> =
            names.iter().map(String::as_str).zip(bodies.iter().map(Vec::as_slice)).collect();

        let archive = build(&entries);
        let parsed = parse(&archive);

        assert_eq!(parsed.len(), 200);
        for (i, entry) in parsed.iter().enumerate() {
            assert_eq!(entry.name, names[i]);
            assert_eq!(entry.data, bodies[i], "entry {i}");
        }
    }

    /// Reading in small pieces must give the same result as reading in one go.
    #[test]
    fn is_insensitive_to_how_the_stream_chunks() {
        struct Dribble<R> {
            inner: R,
            chunk: usize,
        }
        impl<R: std::io::Read> std::io::Read for Dribble<R> {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let take = buf.len().min(self.chunk);
                self.inner.read(&mut buf[..take])
            }
        }

        let archive = build(&[("a.txt", b"first"), ("b.txt", &vec![b'y'; 1500])]);
        for chunk in [1usize, 7, 100, 512, 4096] {
            let reader = Dribble { inner: std::io::Cursor::new(archive.clone()), chunk };
            let entries: Vec<RawEntry> = TarEntries::new(reader).map(|e| e.unwrap()).collect();
            assert_eq!(entries.len(), 2, "chunk {chunk}");
            assert_eq!(entries[1].data.len(), 1500, "chunk {chunk}");
        }
    }

    #[test]
    fn matches_the_tar_crate_on_the_bundled_shards() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
        for name in ["mpdata.tar", "tendata.tar", "testgz.tar", "ixtest.tar"] {
            let bytes = std::fs::read(root.join(name)).unwrap();

            let mine: Vec<(String, usize)> = parse(&bytes).into_iter().map(|e| (e.name, e.data.len())).collect();

            let mut archive = tar::Archive::new(std::io::Cursor::new(&bytes));
            let theirs: Vec<(String, usize)> = archive
                .entries()
                .unwrap()
                .filter_map(|e| {
                    let mut e = e.unwrap();
                    if !e.header().entry_type().is_file() {
                        return None;
                    }
                    let path = e.path().unwrap().to_string_lossy().into_owned();
                    let mut body = Vec::new();
                    std::io::Read::read_to_end(&mut e, &mut body).unwrap();
                    Some((path, body.len()))
                })
                .collect();

            assert_eq!(mine, theirs, "{name} parsed differently");
            assert!(!mine.is_empty(), "{name} should not be empty");
        }
    }

    #[test]
    fn round_trips_the_bundled_shard_contents_byte_for_byte() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/mpdata.tar");
        let bytes = std::fs::read(&path).unwrap();

        let mut archive = tar::Archive::new(std::io::Cursor::new(&bytes));
        let theirs: Vec<Vec<u8>> = archive
            .entries()
            .unwrap()
            .map(|e| {
                let mut e = e.unwrap();
                let mut body = Vec::new();
                std::io::Read::read_to_end(&mut e, &mut body).unwrap();
                body
            })
            .collect();

        let mine: Vec<Vec<u8>> = parse(&bytes).into_iter().map(|e| e.data).collect();
        assert_eq!(mine, theirs);
    }

    #[test]
    fn writing_then_parsing_survives_gzip() {
        let archive = build(&[("a.txt", b"gzipped")]);
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&archive).unwrap();
        let compressed = encoder.finish().unwrap();

        let decoded = flate2::read::MultiGzDecoder::new(std::io::Cursor::new(compressed));
        let entries: Vec<RawEntry> = TarEntries::new(decoded).map(|e| e.unwrap()).collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].data, b"gzipped");
    }
}
