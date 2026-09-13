//! Moving values between Rust and Python.
//!
//! A Python caller expects the objects the reference implementation hands out:
//! `bytes` for undecoded fields, `str` for text, `int` for a class label,
//! `dict`/`list` for JSON, and a NumPy array for anything numeric. This module
//! is the one place that mapping is written down.

use numpy::ndarray::ArrayD;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use webdataset::core::tensor::{DType, Tensor};
use webdataset::{Error, Sample, Value};

/// Convert a decoded value into the Python object a caller expects.
pub fn value_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Int(i) => i.into_pyobject(py)?.into_any(),
        Value::Float(f) => f.into_pyobject(py)?.into_any(),
        Value::Text(s) => s.into_pyobject(py)?.into_any(),
        Value::Bytes(b) => PyBytes::new(py, b).into_any(),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Map(map) => {
            let dict = PyDict::new(py);
            for (name, item) in map {
                dict.set_item(name, value_to_py(py, item)?)?;
            }
            dict.into_any()
        }
        Value::Tensor(t) => tensor_to_py(py, t)?,
        // An image reaches Python as the array `np.asarray` would give, which
        // is what the reference implementation's numeric imagespecs produce.
        // The Python layer wraps it back up for the `pil` specs.
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "cannot convert {} to a Python object",
                other.type_name()
            )));
        }
    })
}

/// Convert a tensor into a NumPy array of the matching dtype and shape.
pub fn tensor_to_py<'py>(py: Python<'py>, tensor: &Tensor) -> PyResult<Bound<'py, PyAny>> {
    let shape = tensor.shape().to_vec();

    /// Reinterpret the native-endian element bytes as a typed array.
    macro_rules! build {
        ($ty:ty, $size:expr, $from:path) => {{
            let values: Vec<$ty> =
                tensor.data().chunks_exact($size).map(|c| $from(c.try_into().expect("sized chunk"))).collect();
            let array = ArrayD::from_shape_vec(shape, values)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            array.to_pyarray(py).into_any()
        }};
    }

    Ok(match tensor.dtype() {
        DType::U8 => PyArrayDyn::<u8>::from_owned_array(
            py,
            ArrayD::from_shape_vec(shape, tensor.data().to_vec())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        )
        .into_any(),
        DType::I8 => build!(i8, 1, i8_from),
        DType::U16 => build!(u16, 2, u16::from_ne_bytes),
        DType::I16 => build!(i16, 2, i16::from_ne_bytes),
        DType::U32 => build!(u32, 4, u32::from_ne_bytes),
        DType::I32 => build!(i32, 4, i32::from_ne_bytes),
        DType::F32 => build!(f32, 4, f32::from_ne_bytes),
        DType::U64 => build!(u64, 8, u64::from_ne_bytes),
        DType::I64 => build!(i64, 8, i64::from_ne_bytes),
        DType::F64 => build!(f64, 8, f64::from_ne_bytes),
        // NumPy has float16, but Rust has no stable f16 to convert through, so
        // it is widened rather than refused.
        DType::F16 => {
            let values: Vec<f32> = tensor.to_f64_vec().into_iter().map(|v| v as f32).collect();
            let array = ArrayD::from_shape_vec(shape, values)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            array.to_pyarray(py).into_any()
        }
    })
}

/// `i8::from_ne_bytes` takes a one-element array; give the macro a uniform shape.
fn i8_from(bytes: [u8; 1]) -> i8 {
    bytes[0] as i8
}

/// Convert a sample into the `dict` the Python API yields.
pub fn sample_to_py<'py>(py: Python<'py>, sample: &Sample) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in sample {
        dict.set_item(name, value_to_py(py, value)?)?;
    }
    Ok(dict)
}

/// Convert a Python object into a value that can be written to a shard.
pub fn py_to_value(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

    if object.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(bytes) = object.cast::<PyBytes>() {
        return Ok(Value::Bytes(bytes.as_bytes().to_vec().into()));
    }
    if let Ok(text) = object.cast::<PyString>() {
        return Ok(Value::Text(text.extract()?));
    }
    // `bool` is checked before `int`, since in Python it is one.
    if let Ok(flag) = object.cast::<PyBool>() {
        return Ok(Value::Bool(flag.is_true()));
    }
    if let Ok(integer) = object.cast::<PyInt>() {
        return Ok(Value::Int(integer.extract()?));
    }
    if let Ok(number) = object.cast::<PyFloat>() {
        return Ok(Value::Float(number.extract()?));
    }
    if let Ok(dict) = object.cast::<PyDict>() {
        let mut map = webdataset::core::fields::Fields::default();
        for (key, item) in dict {
            map.insert(key.extract::<String>()?, py_to_value(&item)?);
        }
        return Ok(Value::Map(map));
    }
    if let Ok(list) = object.cast::<PyList>() {
        return Ok(Value::List(list.iter().map(|item| py_to_value(&item)).collect::<PyResult<_>>()?));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "cannot store a {} in a shard; use bytes, str, int, float, list or dict",
        object.get_type().name()?
    )))
}

/// Convert a Python `dict` into a sample.
pub fn py_to_sample(dict: &Bound<'_, PyDict>) -> PyResult<Sample> {
    let mut sample = Sample::new();
    for (key, value) in dict {
        sample.insert(key.extract::<String>()?, py_to_value(&value)?);
    }
    Ok(sample)
}

/// Present a library error as the Python exception a caller would expect.
pub fn to_py_error(error: Error) -> PyErr {
    match &error {
        Error::Io(_) => pyo3::exceptions::PyIOError::new_err(error.to_string()),
        Error::MissingKey { .. } => pyo3::exceptions::PyKeyError::new_err(error.to_string()),
        Error::Unsupported(_) => pyo3::exceptions::PyNotImplementedError::new_err(error.to_string()),
        // The reference raises ValueError for a duplicate field, and callers
        // catch it, so match rather than settling for RuntimeError.
        Error::Value(_) | Error::Format(_) | Error::DuplicateKey { .. } => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        _ => pyo3::exceptions::PyRuntimeError::new_err(error.to_string()),
    }
}
