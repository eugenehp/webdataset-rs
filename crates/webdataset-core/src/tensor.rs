//! A minimal dense n-dimensional array.
//!
//! WebDataset stores tensors in two formats: NumPy's `.npy` and the 8-byte
//! aligned `.ten` ("tenbin") format. Both are described by an element type, a
//! shape, and a blob of C-ordered, native-endian element data, which is exactly
//! what [`Tensor`] holds. Keeping the representation this simple means the
//! crate does not force a particular array library on its users while still
//! allowing zero-copy round trips through the archive formats.

use bytes::Bytes;

use crate::error::{Error, Result};
use crate::prelude::*;

/// The element type of a [`Tensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 binary16. Read and widened to `f64`; never produced by this
    /// crate, since Rust has no stable 16-bit float.
    F16,
    /// IEEE-754 binary32, NumPy's `float32`.
    F32,
    /// IEEE-754 binary64, NumPy's `float64`.
    F64,
    /// Signed 8-bit.
    I8,
    /// Signed 16-bit.
    I16,
    /// Signed 32-bit.
    I32,
    /// Signed 64-bit.
    I64,
    /// Unsigned 8-bit, which is what a decoded image is made of.
    U8,
    /// Unsigned 16-bit.
    U16,
    /// Unsigned 32-bit.
    U32,
    /// Unsigned 64-bit.
    U64,
}

impl DType {
    /// Size of one element in bytes.
    pub const fn size(self) -> usize {
        match self {
            DType::I8 | DType::U8 => 1,
            DType::F16 | DType::I16 | DType::U16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
        }
    }

    /// The NumPy long name, e.g. `"float32"`.
    pub const fn long_name(self) -> &'static str {
        match self {
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
        }
    }

    /// The two character NumPy short name, e.g. `"f4"`.
    pub const fn short_name(self) -> &'static str {
        match self {
            DType::F16 => "f2",
            DType::F32 => "f4",
            DType::F64 => "f8",
            DType::I8 => "i1",
            DType::I16 => "i2",
            DType::I32 => "i4",
            DType::I64 => "i8",
            DType::U8 => "u1",
            DType::U16 => "u2",
            DType::U32 => "u4",
            DType::U64 => "u8",
        }
    }

    /// Parse either the long (`"float32"`) or short (`"f4"`) NumPy name.
    pub fn parse(name: &str) -> Result<DType> {
        let dt = match name {
            "float16" | "f2" => DType::F16,
            "float32" | "f4" => DType::F32,
            "float64" | "f8" => DType::F64,
            "int8" | "i1" => DType::I8,
            "int16" | "i2" => DType::I16,
            "int32" | "i4" => DType::I32,
            "int64" | "i8" => DType::I64,
            "uint8" | "u1" => DType::U8,
            "uint16" | "u2" => DType::U16,
            "uint32" | "u4" => DType::U32,
            "uint64" | "u8" => DType::U64,
            other => return Err(Error::unsupported(format!("dtype {other}"))),
        };
        Ok(dt)
    }

    /// Whether this is a floating point type.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::F64)
    }
}

/// A dense, C-ordered array of numbers.
///
/// Element bytes are stored in native endianness, matching what NumPy and the
/// tenbin format write on the same machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    dtype: DType,
    shape: Vec<usize>,
    data: Bytes,
}

impl Tensor {
    /// Build a tensor from raw native-endian element bytes.
    ///
    /// Fails when `data` does not hold exactly `shape.product() * dtype.size()`
    /// bytes.
    pub fn new(dtype: DType, shape: Vec<usize>, data: impl Into<Bytes>) -> Result<Tensor> {
        let data = data.into();
        let expected = shape.iter().product::<usize>() * dtype.size();
        if data.len() != expected {
            return Err(Error::format(format!(
                "tensor data is {} bytes but shape {shape:?} of {} needs {expected}",
                data.len(),
                dtype.long_name()
            )));
        }
        Ok(Tensor { dtype, shape, data })
    }

    /// Build a one-dimensional `f32` tensor.
    pub fn from_f32(values: &[f32]) -> Tensor {
        Self::from_f32_shaped(values, vec![values.len()])
    }

    /// Build an `f32` tensor with an explicit shape.
    ///
    /// # Panics
    /// Panics if `shape` does not describe exactly `values.len()` elements.
    pub fn from_f32_shaped(values: &[f32], shape: Vec<usize>) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "shape does not match value count");
        let mut data = Vec::with_capacity(values.len() * 4);
        for v in values {
            data.extend_from_slice(&v.to_ne_bytes());
        }
        Tensor { dtype: DType::F32, shape, data: data.into() }
    }

    /// Build a `u8` tensor with an explicit shape.
    ///
    /// # Panics
    /// Panics if `shape` does not describe exactly `values.len()` elements.
    pub fn from_u8_shaped(values: Vec<u8>, shape: Vec<usize>) -> Tensor {
        assert_eq!(shape.iter().product::<usize>(), values.len(), "shape does not match value count");
        Tensor { dtype: DType::U8, shape, data: values.into() }
    }

    /// The element type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The shape, outermost dimension first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The raw native-endian element bytes.
    pub fn data(&self) -> &Bytes {
        &self.data
    }

    /// Consume the tensor and return its raw element bytes.
    pub fn into_data(self) -> Bytes {
        self.data
    }

    /// Change the shape, keeping the element data.
    ///
    /// Fails when the new shape describes a different number of elements.
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Tensor> {
        Tensor::new(self.dtype, shape, self.data.clone())
    }

    /// Copy the elements out as `f64`, whatever the stored type.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        let d = &self.data[..];
        for i in 0..n {
            out.push(self.element_at(d, i));
        }
        out
    }

    /// Copy the elements out as `f32`, whatever the stored type.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.to_f64_vec().into_iter().map(|v| v as f32).collect()
    }

    /// Read a single element by flat index, widened to `f64`.
    ///
    /// # Panics
    /// Panics if `index` is out of bounds.
    pub fn get(&self, index: usize) -> f64 {
        assert!(index < self.numel(), "index {index} out of bounds");
        self.element_at(&self.data[..], index)
    }

    fn element_at(&self, d: &[u8], i: usize) -> f64 {
        let s = self.dtype.size();
        let b = &d[i * s..(i + 1) * s];
        match self.dtype {
            DType::U8 => b[0] as f64,
            DType::I8 => b[0] as i8 as f64,
            DType::U16 => u16::from_ne_bytes([b[0], b[1]]) as f64,
            DType::I16 => i16::from_ne_bytes([b[0], b[1]]) as f64,
            DType::F16 => f16_to_f64(u16::from_ne_bytes([b[0], b[1]])),
            DType::U32 => u32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64,
            DType::I32 => i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64,
            DType::F32 => f32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as f64,
            DType::U64 => u64::from_ne_bytes(b.try_into().expect("8 bytes")) as f64,
            DType::I64 => i64::from_ne_bytes(b.try_into().expect("8 bytes")) as f64,
            DType::F64 => f64::from_ne_bytes(b.try_into().expect("8 bytes")),
        }
    }
}

/// Widen an IEEE-754 binary16 bit pattern to `f64`.
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0f64 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f64;
    match exponent {
        0 => sign * mantissa * exp2(-24),
        0x1f if mantissa == 0.0 => sign * f64::INFINITY,
        0x1f => f64::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * exp2(exponent - 15),
    }
}

/// Two raised to `n`, computed without `std`'s float intrinsics.
///
/// Powers of two are exactly representable, and binary16 only ever needs
/// exponents in `-24..=16`, so repeated halving or doubling is exact.
fn exp2(n: i32) -> f64 {
    let mut out = 1.0f64;
    for _ in 0..n.abs() {
        if n >= 0 {
            out *= 2.0;
        } else {
            out /= 2.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_f32_values() {
        let t = Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.to_f64_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.get(4), 5.0);
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let err = Tensor::new(DType::F32, vec![3], vec![0u8; 8]);
        assert!(err.is_err());
    }

    #[test]
    fn reshapes_without_copying_semantics() {
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let r = t.reshape(vec![2, 2]).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert!(t.reshape(vec![3, 3]).is_err());
    }

    #[test]
    fn parses_dtype_names() {
        assert_eq!(DType::parse("float32").unwrap(), DType::F32);
        assert_eq!(DType::parse("f4").unwrap(), DType::F32);
        assert_eq!(DType::parse("u1").unwrap(), DType::U8);
        assert!(DType::parse("complex64").is_err());
    }
}
