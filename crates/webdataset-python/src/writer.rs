//! Writing shards from Python.

use std::sync::{Arc, Mutex};

use numpy::{PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use webdataset::core::tensor::{DType, Tensor};
use webdataset::{DefaultEncoder, Value};

use crate::convert::to_py_error;

/// Convert a NumPy array into a tensor, without copying more than once.
pub fn numpy_to_tensor(array: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    /// Try one dtype, and fall through if the array is not of that type.
    macro_rules! attempt {
        ($ty:ty, $dtype:expr) => {
            if let Ok(typed) = array.cast::<PyArrayDyn<$ty>>() {
                let shape = typed.shape().to_vec();
                let readonly = typed.try_readonly()?;
                let values = readonly.as_slice()?;
                let mut bytes = Vec::with_capacity(std::mem::size_of::<$ty>() * values.len());
                for value in values {
                    bytes.extend_from_slice(&value.to_ne_bytes());
                }
                return Tensor::new($dtype, shape, bytes).map_err(to_py_error);
            }
        };
    }

    attempt!(f32, DType::F32);
    attempt!(f64, DType::F64);
    attempt!(u8, DType::U8);
    attempt!(i8, DType::I8);
    attempt!(u16, DType::U16);
    attempt!(i16, DType::I16);
    attempt!(u32, DType::U32);
    attempt!(i32, DType::I32);
    attempt!(u64, DType::U64);
    attempt!(i64, DType::I64);

    Err(pyo3::exceptions::PyTypeError::new_err(
        "unsupported array dtype; expected one of float32, float64, int8..int64, uint8..uint64",
    ))
}

/// Turn a Python object into a value, understanding NumPy arrays too.
fn to_value(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    if object.getattr("__array_interface__").is_ok() || object.get_type().name()? == "ndarray" {
        if let Ok(tensor) = numpy_to_tensor(object) {
            return Ok(Value::Tensor(tensor));
        }
    }
    crate::convert::py_to_value(object)
}

/// Convert a Python `dict` into a sample, understanding NumPy arrays.
fn to_sample(dict: &Bound<'_, PyDict>) -> PyResult<webdataset::Sample> {
    let mut sample = webdataset::Sample::new();
    for (key, value) in dict {
        sample.insert(key.extract::<String>()?, to_value(&value)?);
    }
    Ok(sample)
}

/// Writes samples into a single tar archive.
#[pyclass(module = "webdataset._native")]
pub struct TarWriter {
    /// A `#[pyclass]` has to be `Sync`, and a writer is not.
    inner: Mutex<Option<webdataset::TarWriter>>,
    written: Mutex<usize>,
}

#[pymethods]
impl TarWriter {
    /// Create the archive at `path`, choosing compression from its name.
    #[new]
    #[pyo3(signature = (path, *, encoder = true, mtime = None))]
    fn new(path: &str, encoder: bool, mtime: Option<u64>) -> PyResult<TarWriter> {
        let mut writer = webdataset::TarWriter::create(path).map_err(to_py_error)?.with_mtime(mtime);
        if encoder {
            writer = writer.with_encoder(Arc::new(DefaultEncoder::new()));
        }
        Ok(TarWriter { inner: Mutex::new(Some(writer)), written: Mutex::new(0) })
    }

    /// Append a sample.
    fn write(&self, sample: &Bound<'_, PyDict>) -> PyResult<u64> {
        let converted = to_sample(sample)?;
        let mut guard = self.inner.lock().expect("writer lock");
        let Some(writer) = guard.as_mut() else {
            return Err(pyo3::exceptions::PyValueError::new_err("the writer is closed"));
        };
        let size = writer.write(&converted).map_err(to_py_error)?;
        *self.written.lock().expect("counter lock") += 1;
        Ok(size)
    }

    /// Finish the archive.
    fn close(&self) -> PyResult<()> {
        match self.inner.lock().expect("writer lock").take() {
            Some(writer) => writer.close().map_err(to_py_error),
            None => Ok(()),
        }
    }

    /// How many samples have been written.
    #[getter]
    fn count(&self) -> usize {
        *self.written.lock().expect("counter lock")
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Writes samples across a numbered series of shards.
#[pyclass(module = "webdataset._native")]
pub struct ShardWriter {
    /// A `#[pyclass]` has to be `Sync`, and a writer is not.
    inner: Mutex<Option<webdataset::ShardWriter>>,
}

#[pymethods]
impl ShardWriter {
    /// Write shards named by a `printf`-style pattern such as `out-%06d.tar`.
    #[new]
    #[pyo3(signature = (pattern, *, maxcount = 100_000, maxsize = 3_000_000_000, encoder = true, start_shard = 0))]
    fn new(pattern: &str, maxcount: usize, maxsize: u64, encoder: bool, start_shard: usize) -> PyResult<ShardWriter> {
        let mut writer = webdataset::ShardWriter::new(pattern)
            .map_err(to_py_error)?
            .with_max_count(maxcount)
            .with_max_size(maxsize)
            .with_start_shard(start_shard);
        if encoder {
            writer = writer.with_encoder(Arc::new(DefaultEncoder::new()));
        }
        Ok(ShardWriter { inner: Mutex::new(Some(writer)) })
    }

    /// Append a sample, rolling over to a new shard if the limits are reached.
    fn write(&self, sample: &Bound<'_, PyDict>) -> PyResult<()> {
        let converted = to_sample(sample)?;
        let mut guard = self.inner.lock().expect("writer lock");
        let Some(writer) = guard.as_mut() else {
            return Err(pyo3::exceptions::PyValueError::new_err("the writer is closed"));
        };
        writer.write(&converted).map_err(to_py_error)
    }

    /// Finish the current shard and start the next one.
    fn next_stream(&self) -> PyResult<()> {
        match self.inner.lock().expect("writer lock").as_mut() {
            Some(writer) => writer.next_shard().map_err(to_py_error),
            None => Ok(()),
        }
    }

    /// Finish the last shard.
    fn close(&self) -> PyResult<()> {
        match self.inner.lock().expect("writer lock").take() {
            Some(writer) => writer.close().map_err(to_py_error),
            None => Ok(()),
        }
    }

    /// How many samples have been written across all shards.
    #[getter]
    fn total(&self) -> usize {
        self.inner.lock().expect("writer lock").as_ref().map(|w| w.total()).unwrap_or(0)
    }

    /// How many shards have been started.
    #[getter]
    fn shard(&self) -> usize {
        self.inner.lock().expect("writer lock").as_ref().map(|w| w.shard_count()).unwrap_or(0)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}
