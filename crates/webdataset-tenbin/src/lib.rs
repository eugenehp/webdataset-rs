//! The tenbin (`.ten`) binary tensor format.
//!
//! Tenbin is an 8-byte aligned encoding for lists of dense arrays. Because the
//! payload is aligned and stored in native byte order, a decoder can hand the
//! bytes straight to a compute kernel or an RDMA transfer without copying.
//!
//! A file is a sequence of chunks:
//!
//! ```text
//! magic     8 bytes, "~TenBin~"
//! length    8 bytes, native-endian i64, the unpadded length of the payload
//! payload   `length` bytes, zero padded up to a multiple of 64
//! ```
//!
//! Each array contributes two chunks: a header and its element data. The
//! header is itself a run of 8-byte fields:
//!
//! ```text
//! dtype     the short NumPy name ("f4", "i8", ...), NUL padded to 8 bytes
//! info      a free-form 8 byte label, NUL padded
//! ndim      native-endian i64
//! shape     `ndim` native-endian i64 values
//! ```
//!
//! ```
//! use webdataset_core::Tensor;
//! use webdataset_tenbin::{decode_buffer, encode_buffer};
//!
//! let tensors = vec![Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])];
//! let encoded = encode_buffer(&tensors);
//! assert_eq!(decode_buffer(&encoded).unwrap(), tensors);
//! ```

#![doc(html_root_url = "https://docs.rs/webdataset-tenbin/0.0.1")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::format;
use alloc::string::{String, ToString};
#[allow(unused_imports)]
use alloc::vec;
use alloc::vec::Vec;

use webdataset_core::error::{Error, Result};
use webdataset_core::tensor::{DType, Tensor};

/// The 8 byte marker that starts every chunk.
pub const MAGIC: &[u8; 8] = b"~TenBin~";

/// Chunk payloads are padded up to a multiple of this many bytes.
pub const ALIGNMENT: usize = 64;

/// An array together with its 8 byte info label.
#[derive(Debug, Clone, PartialEq)]
pub struct Labelled {
    /// The array itself.
    pub tensor: Tensor,
    /// The label stored alongside it; at most 8 ASCII bytes.
    pub info: String,
}

/// Round `n` up to the next multiple of `k`.
fn roundup(n: usize, k: usize) -> usize {
    k * n.div_ceil(k)
}

/// Encode a string into the fixed 8 byte field the format uses.
fn field(text: &str) -> Result<[u8; 8]> {
    let bytes = text.as_bytes();
    if bytes.len() > 8 {
        return Err(Error::value(format!("tenbin label {text:?} is longer than 8 bytes")));
    }
    let mut out = [0u8; 8];
    out[..bytes.len()].copy_from_slice(bytes);
    Ok(out)
}

/// Decode one of the fixed 8 byte string fields.
fn unfield(bytes: &[u8]) -> Result<String> {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    Ok(core::str::from_utf8(&bytes[..end])?.to_string())
}

/// Encode a list of arrays into a single tenbin byte string.
pub fn encode_buffer(tensors: &[Tensor]) -> Vec<u8> {
    encode_labelled(&tensors.iter().cloned().map(|tensor| Labelled { tensor, info: String::new() }).collect::<Vec<_>>())
        .expect("empty labels always fit")
}

/// Encode a list of labelled arrays into a single tenbin byte string.
pub fn encode_labelled(items: &[Labelled]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for item in items {
        push_chunk(&mut out, &encode_header(item)?);
        push_chunk(&mut out, item.tensor.data());
    }
    Ok(out)
}

/// Decode a tenbin byte string into a list of arrays.
pub fn decode_buffer(data: &[u8]) -> Result<Vec<Tensor>> {
    Ok(decode_labelled(data)?.into_iter().map(|l| l.tensor).collect())
}

/// Decode a tenbin byte string, keeping each array's label.
pub fn decode_labelled(data: &[u8]) -> Result<Vec<Labelled>> {
    let chunks = decode_chunks(data)?;
    if chunks.len() % 2 != 0 {
        return Err(Error::format("tenbin buffer ends with a header but no data"));
    }
    let mut out = Vec::with_capacity(chunks.len() / 2);
    for pair in chunks.chunks_exact(2) {
        out.push(decode_array(pair[0], pair[1])?);
    }
    Ok(out)
}

/// Split a tenbin byte string into its raw chunk payloads.
fn decode_chunks(data: &[u8]) -> Result<Vec<&[u8]>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        if data.len() < offset + 16 {
            return Err(Error::format("truncated tenbin chunk header"));
        }
        if &data[offset..offset + 8] != MAGIC {
            return Err(Error::format(format!("tenbin magic mismatch at offset {offset}")));
        }
        offset += 8;
        let length = i64::from_ne_bytes(data[offset..offset + 8].try_into().expect("8 bytes"));
        offset += 8;
        if length < 0 {
            return Err(Error::format("negative tenbin chunk length"));
        }
        let length = length as usize;
        if data.len() < offset + length {
            return Err(Error::format("truncated tenbin chunk payload"));
        }
        out.push(&data[offset..offset + length]);
        offset += roundup(length, ALIGNMENT);
    }
    Ok(out)
}

/// Interpret a header chunk and its data chunk as one array.
fn decode_array(header: &[u8], data: &[u8]) -> Result<Labelled> {
    if header.len() < 24 {
        return Err(Error::format("tenbin header is too short"));
    }
    let dtype = DType::parse(&unfield(&header[0..8])?)?;
    let info = unfield(&header[8..16])?;
    let ndim = i64::from_ne_bytes(header[16..24].try_into().expect("8 bytes"));
    if !(0..=10).contains(&ndim) {
        return Err(Error::format(format!("tenbin rank {ndim} is out of range")));
    }
    let ndim = ndim as usize;
    if header.len() < 24 + ndim * 8 {
        return Err(Error::format("tenbin header is shorter than its rank implies"));
    }
    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let at = 24 + i * 8;
        let dim = i64::from_ne_bytes(header[at..at + 8].try_into().expect("8 bytes"));
        if dim < 0 {
            return Err(Error::format("negative tenbin dimension"));
        }
        shape.push(dim as usize);
    }
    Ok(Labelled { tensor: Tensor::new(dtype, shape, data.to_vec())?, info })
}

/// Build the header chunk for one array.
fn encode_header(item: &Labelled) -> Result<Vec<u8>> {
    let t = &item.tensor;
    if t.ndim() >= 10 {
        return Err(Error::value(format!("tenbin supports at most 9 dimensions, got {}", t.ndim())));
    }
    let mut header = Vec::with_capacity(24 + t.ndim() * 8);
    header.extend_from_slice(&field(t.dtype().short_name())?);
    header.extend_from_slice(&field(&item.info)?);
    header.extend_from_slice(&(t.ndim() as i64).to_ne_bytes());
    for dim in t.shape() {
        header.extend_from_slice(&(*dim as i64).to_ne_bytes());
    }
    Ok(header)
}

/// Append one chunk to `out`: magic, length, payload, padding.
pub fn push_chunk(out: &mut Vec<u8>, payload: &[u8]) {
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(payload.len() as i64).to_ne_bytes());
    out.extend_from_slice(payload);
    out.resize(out.len() + (roundup(payload.len(), ALIGNMENT) - payload.len()), 0);
}

/// Write one chunk: magic, length, payload, padding.
#[cfg(feature = "std")]
pub fn write_chunk(out: &mut impl std::io::Write, payload: &[u8]) -> Result<()> {
    out.write_all(MAGIC)?;
    out.write_all(&(payload.len() as i64).to_ne_bytes())?;
    out.write_all(payload)?;
    let padding = roundup(payload.len(), ALIGNMENT) - payload.len();
    if padding > 0 {
        out.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}

/// Read one chunk, returning `None` at a clean end of stream.
#[cfg(feature = "std")]
pub fn read_chunk(stream: &mut impl std::io::Read) -> Result<Option<Vec<u8>>> {
    let mut magic = [0u8; 8];
    if !read_exact_or_eof(stream, &mut magic)? {
        return Ok(None);
    }
    if &magic != MAGIC {
        return Err(Error::format("tenbin magic mismatch"));
    }
    let mut length = [0u8; 8];
    stream.read_exact(&mut length)?;
    let length = i64::from_ne_bytes(length);
    if length < 0 {
        return Err(Error::format("negative tenbin chunk length"));
    }
    let length = length as usize;

    // Grow with the bytes that actually arrive rather than reserving the
    // declared length. The length comes off the wire, so a crafted chunk can
    // name a size no machine can allocate; reserving it up front aborts the
    // process instead of returning the error the caller could handle.
    const MAX_PREALLOC: usize = 1 << 20;
    const CHUNK: usize = 1 << 16;
    let mut payload = Vec::with_capacity(length.min(MAX_PREALLOC));
    while payload.len() < length {
        let start = payload.len();
        payload.resize(start + CHUNK.min(length - start), 0);
        stream.read_exact(&mut payload[start..])?;
    }
    let padding = roundup(length, ALIGNMENT) - length;
    if padding > 0 {
        let mut skip = vec![0u8; padding];
        stream.read_exact(&mut skip)?;
    }
    Ok(Some(payload))
}

/// Write a list of arrays to a stream.
#[cfg(feature = "std")]
pub fn write(out: &mut impl std::io::Write, tensors: &[Tensor]) -> Result<()> {
    let items: Vec<Labelled> = tensors.iter().cloned().map(|tensor| Labelled { tensor, info: String::new() }).collect();
    write_labelled(out, &items)
}

/// Write a list of labelled arrays to a stream.
#[cfg(feature = "std")]
pub fn write_labelled(out: &mut impl std::io::Write, items: &[Labelled]) -> Result<()> {
    for item in items {
        write_chunk(out, &encode_header(item)?)?;
        write_chunk(out, item.tensor.data())?;
    }
    Ok(())
}

/// Read a list of arrays from a stream.
#[cfg(feature = "std")]
pub fn read(stream: &mut impl std::io::Read) -> Result<Vec<Tensor>> {
    Ok(read_labelled(stream)?.into_iter().map(|l| l.tensor).collect())
}

/// Read a list of labelled arrays from a stream.
#[cfg(feature = "std")]
pub fn read_labelled(stream: &mut impl std::io::Read) -> Result<Vec<Labelled>> {
    let mut out = Vec::new();
    loop {
        let Some(header) = read_chunk(stream)? else {
            return Ok(out);
        };
        let data = read_chunk(stream)?.ok_or_else(|| Error::format("tenbin stream ends after a header"))?;
        out.push(decode_array(&header, &data)?);
    }
}

/// Write arrays to a `.ten` file.
#[cfg(feature = "std")]
pub fn save(path: impl AsRef<std::path::Path>, tensors: &[Tensor]) -> Result<()> {
    use std::io::Write;
    let path = path.as_ref();
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    write(&mut file, tensors)?;
    file.flush()?;
    Ok(())
}

/// Read arrays from a `.ten` file.
#[cfg(feature = "std")]
pub fn load(path: impl AsRef<std::path::Path>) -> Result<Vec<Tensor>> {
    let mut file = std::io::BufReader::new(std::fs::File::open(path.as_ref())?);
    read(&mut file)
}

/// Fill `buf`, reporting `false` when the stream ended before any byte arrived.
#[cfg(feature = "std")]
fn read_exact_or_eof(stream: &mut impl std::io::Read, buf: &mut [u8]) -> Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match stream.read(&mut buf[filled..])? {
            0 if filled == 0 => return Ok(false),
            0 => return Err(Error::format("truncated tenbin chunk")),
            n => filled += n,
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<Tensor> {
        vec![
            Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            Tensor::from_u8_shaped(vec![9, 8, 7], vec![3]),
        ]
    }

    #[test]
    fn round_trips_through_a_buffer() {
        let tensors = sample();
        let encoded = encode_buffer(&tensors);
        assert_eq!(&encoded[..8], MAGIC);
        assert_eq!(decode_buffer(&encoded).unwrap(), tensors);
    }

    #[test]
    fn pads_every_chunk_to_the_alignment() {
        let encoded = encode_buffer(&[Tensor::from_u8_shaped(vec![1, 2, 3], vec![3])]);
        // header chunk: 16 + 64, data chunk: 16 + 64
        assert_eq!(encoded.len(), 160);
        assert_eq!(&encoded[80..88], MAGIC, "the second chunk starts right after the padding");
    }

    #[cfg(feature = "std")]
    #[test]
    fn round_trips_through_a_stream() {
        let tensors = sample();
        let mut buf = Vec::new();
        write(&mut buf, &tensors).unwrap();
        assert_eq!(read(&mut &buf[..]).unwrap(), tensors);
    }

    #[test]
    fn keeps_labels() {
        let items = vec![Labelled { tensor: Tensor::from_f32(&[1.0]), info: "img".into() }];
        let encoded = encode_labelled(&items).unwrap();
        assert_eq!(decode_labelled(&encoded).unwrap(), items);
    }

    #[test]
    fn rejects_oversized_labels() {
        let items = vec![Labelled { tensor: Tensor::from_f32(&[1.0]), info: "much too long".into() }];
        assert!(encode_labelled(&items).is_err());
    }

    #[test]
    fn rejects_corrupt_input() {
        let mut encoded = encode_buffer(&sample());
        encoded[2] = b'X';
        assert!(decode_buffer(&encoded).is_err());
        assert!(decode_buffer(&[0u8; 4]).is_err());
        assert_eq!(decode_buffer(&[]).unwrap().len(), 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("tenbin-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.ten");
        let tensors = sample();
        save(&path, &tensors).unwrap();
        assert_eq!(load(&path).unwrap(), tensors);
        std::fs::remove_dir_all(&dir).ok();
    }
}
