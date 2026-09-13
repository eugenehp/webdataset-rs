//! Reading and writing NumPy's `.npy` and `.npz` container formats.
//!
//! Only the dense numeric subset that WebDataset actually stores is supported:
//! C-ordered arrays of the dtypes in [`DType`]. Fortran-ordered arrays,
//! structured dtypes, and object arrays are rejected with a clear error.

use crate::error::{Error, Result};
use crate::prelude::*;
use crate::tensor::{DType, Tensor};

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const ALIGNMENT: usize = 64;

/// Decode a `.npy` byte string into a [`Tensor`].
pub fn from_npy(data: &[u8]) -> Result<Tensor> {
    if data.len() < 10 || &data[..6] != MAGIC {
        return Err(Error::format("not a .npy file (bad magic)"));
    }
    let major = data[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([data[8], data[9]]) as usize, 10),
        2 | 3 => {
            if data.len() < 12 {
                return Err(Error::format("truncated .npy header"));
            }
            (u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize, 12)
        }
        other => return Err(Error::unsupported(format!(".npy version {other}"))),
    };
    let header_end = header_start + header_len;
    if data.len() < header_end {
        return Err(Error::format("truncated .npy header"));
    }
    let header = core::str::from_utf8(&data[header_start..header_end])?;
    let (dtype, big_endian, fortran, shape) = parse_header(header)?;

    let expected = shape.iter().product::<usize>() * dtype.size();
    let body = &data[header_end..];
    if body.len() < expected {
        return Err(Error::format(format!(".npy body is {} bytes but shape {shape:?} needs {expected}", body.len())));
    }
    if fortran {
        return Err(Error::unsupported("Fortran-ordered .npy arrays"));
    }

    let mut body = body[..expected].to_vec();
    if big_endian != cfg!(target_endian = "big") && dtype.size() > 1 {
        for chunk in body.chunks_exact_mut(dtype.size()) {
            chunk.reverse();
        }
    }
    Tensor::new(dtype, shape, body)
}

/// Encode a [`Tensor`] as a version 1.0 `.npy` byte string.
pub fn to_npy(tensor: &Tensor) -> Vec<u8> {
    let mut out = Vec::with_capacity(tensor.data().len() + ALIGNMENT + 16);
    let (header, body) = npy_parts(tensor);
    out.extend_from_slice(&header);
    out.extend_from_slice(body);
    out
}

/// Write a [`Tensor`] to `out` in version 1.0 `.npy` format.
#[cfg(feature = "std")]
pub fn write_npy(out: &mut impl std::io::Write, tensor: &Tensor) -> Result<()> {
    let (header, body) = npy_parts(tensor);
    out.write_all(&header)?;
    out.write_all(body)?;
    Ok(())
}

/// Read a `.npy` array from a stream.
#[cfg(feature = "std")]
pub fn read_npy(stream: &mut impl std::io::Read) -> Result<Tensor> {
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    from_npy(&buf)
}

/// Build the header block and borrow the element data of a `.npy` encoding.
fn npy_parts(tensor: &Tensor) -> (Vec<u8>, &[u8]) {
    let order = if cfg!(target_endian = "big") { '>' } else { '<' };
    let shape = match tensor.shape() {
        [] => "()".to_string(),
        [one] => format!("({one},)"),
        many => format!("({})", many.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header =
        format!("{{'descr': '{order}{}', 'fortran_order': False, 'shape': {shape}, }}", tensor.dtype().short_name());

    // The header is padded so that the start of the data is 64-byte aligned.
    let unpadded = MAGIC.len() + 2 + 2 + header.len() + 1;
    let padding = (ALIGNMENT - unpadded % ALIGNMENT) % ALIGNMENT;
    header.push_str(&" ".repeat(padding));
    header.push('\n');

    let mut block = Vec::with_capacity(unpadded + padding);
    block.extend_from_slice(MAGIC);
    block.extend_from_slice(&[1, 0]);
    block.extend_from_slice(&(header.len() as u16).to_le_bytes());
    block.extend_from_slice(header.as_bytes());
    (block, tensor.data())
}

/// Parse the Python-literal header dict of a `.npy` file.
fn parse_header(header: &str) -> Result<(DType, bool, bool, Vec<usize>)> {
    let descr = extract(header, "descr").ok_or_else(|| Error::format("no descr in .npy header"))?;
    let mut chars = descr.chars();
    let (big_endian, name) = match chars.next() {
        Some('<') | Some('|') | Some('=') => (false, chars.as_str()),
        Some('>') => (true, chars.as_str()),
        _ => (false, descr.as_str()),
    };
    let dtype = DType::parse(name)?;

    let fortran = header.contains("'fortran_order': True");

    let shape_text = header
        .split_once("'shape':")
        .and_then(|(_, rest)| rest.split_once('(').and_then(|(_, r)| r.split_once(')')).map(|(inner, _)| inner))
        .ok_or_else(|| Error::format("no shape in .npy header"))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|_| Error::format(format!("bad shape entry {s:?}"))))
        .collect::<Result<Vec<_>>>()?;

    Ok((dtype, big_endian, fortran, shape))
}

/// Pull `'<key>': '<value>'` out of a Python-literal dict.
fn extract(header: &str, key: &str) -> Option<String> {
    let needle = format!("'{key}':");
    let rest = header.split_once(&needle)?.1;
    let rest = rest.trim_start();
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(body[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_matrix() {
        let t = Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let bytes = to_npy(&t);
        assert_eq!(&bytes[..6], MAGIC);
        // The data must start on a 64 byte boundary.
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % ALIGNMENT, 0);

        let back = from_npy(&bytes).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn round_trips_a_vector_and_a_scalar() {
        for shape in [vec![4], vec![1], vec![2, 2]] {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let t = Tensor::from_f32_shaped(&values, shape);
            assert_eq!(from_npy(&to_npy(&t)).unwrap(), t);
        }
    }

    #[test]
    fn rejects_junk() {
        assert!(from_npy(b"not an npy file at all").is_err());
        assert!(from_npy(b"").is_err());
    }

    #[test]
    fn parses_a_real_numpy_header() {
        let header = "{'descr': '<i8', 'fortran_order': False, 'shape': (3, 4), }";
        let (dtype, big, fortran, shape) = parse_header(header).unwrap();
        assert_eq!(dtype, DType::I64);
        assert!(!big);
        assert!(!fortran);
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn rejects_fortran_order() {
        let bytes = to_npy(&Tensor::from_f32(&[1.0, 2.0]));
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = core::str::from_utf8(&bytes[10..10 + header_len]).unwrap().replace("False", "True ");

        let mut fortran = bytes[..10].to_vec();
        fortran.extend_from_slice(header.as_bytes());
        fortran.extend_from_slice(&bytes[10 + header_len..]);

        assert_eq!(fortran.len(), bytes.len(), "only the flag should change");
        let err = from_npy(&fortran).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{err}");
    }
}
