//! Python bindings for `webdataset-rs`.
//!
//! The Python package built on top of this module presents the same API as the
//! reference `webdataset` library, so existing code runs unchanged. This module
//! is the part that has to be in Rust: reading shards, parsing archives,
//! grouping files into samples, decoding fields, shuffling and batching.
//!
//! Everything here releases the GIL while it works, so a `DataLoader` worker
//! spends its time reading rather than contending.

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use webdataset::core::handlers;
use webdataset::filters::SampleIteratorExt;
use webdataset::{DataPipeline, Decoder, HandlerRef, Sample, Selection, Value, WebDataset};

mod convert;
mod writer;

use convert::{sample_to_py, to_py_error, value_to_py};

/// How a reader should behave, as configured from Python.
#[derive(Debug, Clone, Default)]
struct Config {
    urls: Vec<String>,
    verbatim: bool,
    shardshuffle: Option<usize>,
    shuffle: Option<usize>,
    decode: Option<String>,
    only: Option<Vec<String>>,
    /// Called with each archive member's name; false drops it.
    select_files: Option<Arc<Py<PyAny>>>,
    /// Called with each archive member's name to rewrite it.
    rename_files: Option<Arc<Py<PyAny>>>,
    to_tuple: Option<Vec<String>>,
    batchsize: Option<usize>,
    partial: bool,
    /// Whether a batch is stacked column by column, or left as a list.
    collate: bool,
    resampled: bool,
    repeat: bool,
    epoch: Option<usize>,
    limit: Option<usize>,
    seed: Option<u64>,
    cache_dir: Option<String>,
    handler: String,
    empty_check: bool,
    workersplit: bool,
    nodesplit: bool,
}

/// Translate a handler name into the handler itself.
fn handler_for(name: &str) -> PyResult<HandlerRef> {
    Ok(match name {
        "reraise" => handlers::reraise_exception(),
        "ignore_and_continue" => handlers::ignore_and_continue(),
        "warn_and_continue" => handlers::warn_and_continue(),
        "ignore_and_stop" => handlers::ignore_and_stop(),
        "warn_and_stop" => handlers::warn_and_stop(),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown handler {other:?}; expected reraise, ignore_and_continue, \
                 warn_and_continue, ignore_and_stop or warn_and_stop"
            )));
        }
    })
}

/// Build the pipeline a configuration describes.
fn build(config: &Config) -> PyResult<DataPipeline> {
    let handler = handler_for(&config.handler)?;

    let mut selection = Selection::new();
    // These run inside the reader, before files are grouped into samples, which
    // is where the reference implementation applies them too. Applying them
    // afterwards would look similar until a rename changed a sample's key, at
    // which point the files would no longer regroup.
    if let Some(rename) = config.rename_files.clone() {
        selection = selection.rename(move |name| {
            Python::attach(|py| match rename.call1(py, (name,)).and_then(|out| out.extract::<String>(py)) {
                Ok(renamed) => renamed,
                Err(e) => {
                    // A raising renamer leaves the name alone; the error is
                    // reported rather than swallowed silently.
                    e.restore(py);
                    name.to_string()
                }
            })
        });
    }
    if let Some(select) = config.select_files.clone() {
        selection = selection.select(move |name| {
            Python::attach(|py| match select.call1(py, (name,)).and_then(|out| out.extract::<bool>(py)) {
                Ok(keep) => keep,
                Err(e) => {
                    e.restore(py);
                    true
                }
            })
        });
    }

    let mut builder = if config.verbatim {
        WebDataset::builder_verbatim(config.urls.clone())
    } else {
        WebDataset::builder_from(&config.urls)
    };
    builder = builder
        .selection(selection)
        .handler(handler)
        .empty_check(config.empty_check)
        .resampled(config.resampled)
        .worker_split(config.workersplit)
        .node_split(if config.nodesplit { webdataset::NodeSplit::ByNode } else { webdataset::NodeSplit::Refuse });
    if let Some(bufsize) = config.shardshuffle {
        builder = builder.shard_shuffle(bufsize);
    }
    if let Some(seed) = config.seed {
        builder = builder.seed(seed).deterministic(true);
    }
    if let Some(directory) = &config.cache_dir {
        builder = builder.cache_dir(directory.clone());
    }

    let mut dataset = builder.build().map_err(to_py_error)?;

    if let Some(bufsize) = config.shuffle {
        dataset = dataset.shuffle(bufsize);
    }
    if let Some(spec) = &config.decode {
        dataset = decode_with(dataset, spec, config.only.clone())?;
    }

    let mut pipeline = dataset.into_pipeline();
    if config.repeat {
        pipeline = pipeline.repeat_forever();
    }
    if let Some(n) = config.epoch {
        pipeline = pipeline.with_epoch(n);
    }
    if let Some(n) = config.limit {
        pipeline = pipeline.take(n);
    }
    Ok(pipeline)
}

/// Apply the decoder a spec names.
fn decode_with(dataset: WebDataset, spec: &str, only: Option<Vec<String>>) -> PyResult<WebDataset> {
    let mut decoder = match spec {
        "" | "basic" => Decoder::default(),
        _ => {
            let handler = webdataset::ImageHandler::parse(spec).map_err(to_py_error)?;
            Decoder::new(vec![Arc::new(handler)])
        }
    };
    if let Some(fields) = only {
        decoder = decoder.only(fields);
    }
    Ok(dataset.decode(decoder))
}

/// What a reader hands back on each step.
enum Output {
    /// A whole sample, as a `dict`.
    Sample(Sample),
    /// A tuple of selected fields.
    Row(Vec<Value>),
    /// A list of samples.
    Batch(Vec<Sample>),
    /// A list of tuples.
    RowBatch(Vec<Vec<Value>>),
    /// A batch stacked column by column, as a `dict`.
    Collated(Sample),
    /// A batch of tuples, stacked column by column.
    CollatedRow(Vec<Value>),
}

/// Reads shards and yields samples, batches or tuples.
///
/// This is the fused fast path: shard listing, archive parsing, grouping,
/// decoding, shuffling and batching all happen in Rust, with the GIL released,
/// and only the finished object crosses back into Python.
#[pyclass(module = "webdataset._native")]
pub struct Reader {
    config: Config,
    /// A `#[pyclass]` has to be `Sync`, and an iterator is not; the lock is
    /// what lets the same reader be handed between threads safely.
    iterator: Mutex<Option<Box<dyn Iterator<Item = webdataset::Result<Output>> + Send>>>,
}

#[pymethods]
impl Reader {
    /// Build a reader from a configuration dictionary.
    ///
    /// Taking a dict rather than forty keyword arguments keeps the Python layer
    /// free to grow options without the signature here having to follow.
    #[new]
    fn new(config: &Bound<'_, PyDict>) -> PyResult<Reader> {
        let get_str = |key: &str| -> PyResult<Option<String>> {
            config.get_item(key)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()
        };
        let get_usize = |key: &str| -> PyResult<Option<usize>> {
            config.get_item(key)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()
        };
        let get_u64 = |key: &str| -> PyResult<Option<u64>> {
            config.get_item(key)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()
        };
        let get_bool = |key: &str, default: bool| -> PyResult<bool> {
            Ok(config.get_item(key)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()?.unwrap_or(default))
        };
        let get_strings = |key: &str| -> PyResult<Option<Vec<String>>> {
            config.get_item(key)?.filter(|v| !v.is_none()).map(|v| v.extract()).transpose()
        };

        let urls: Vec<String> = config
            .get_item("urls")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no urls given"))?
            .extract()?;

        let callable = |key: &str| -> PyResult<Option<Arc<Py<PyAny>>>> {
            Ok(config.get_item(key)?.filter(|v| !v.is_none()).map(|v| Arc::new(v.unbind())))
        };

        Ok(Reader {
            config: Config {
                urls,
                verbatim: get_bool("verbatim", false)?,
                shardshuffle: get_usize("shardshuffle")?,
                shuffle: get_usize("shuffle")?,
                decode: get_str("decode")?,
                only: get_strings("only")?,
                select_files: callable("select_files")?,
                rename_files: callable("rename_files")?,
                to_tuple: get_strings("to_tuple")?,
                batchsize: get_usize("batchsize")?,
                partial: get_bool("partial", true)?,
                collate: get_bool("collate", true)?,
                resampled: get_bool("resampled", false)?,
                repeat: get_bool("repeat", false)?,
                epoch: get_usize("epoch")?,
                limit: get_usize("limit")?,
                seed: get_u64("seed")?,
                cache_dir: get_str("cache_dir")?,
                handler: get_str("handler")?.unwrap_or_else(|| "reraise".to_string()),
                empty_check: get_bool("empty_check", true)?,
                workersplit: get_bool("workersplit", true)?,
                nodesplit: get_bool("nodesplit", false)?,
            },
            iterator: Mutex::new(None),
        })
    }

    /// Start an epoch.
    fn __iter__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        let config = slf.config.clone();
        let built = py.detach(|| make_iterator(&config))?;
        *slf.iterator.lock().expect("reader lock") = Some(built);
        Ok(slf)
    }

    /// Produce the next sample, batch or tuple.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        // Reading and decoding happen without the GIL; only building the
        // Python object below needs it.
        let item = py.detach(|| {
            let mut guard = self.iterator.lock().expect("reader lock");
            match guard.as_mut() {
                Some(iterator) => Ok(iterator.next()),
                None => Err(()),
            }
        });
        match item {
            Err(()) => Err(pyo3::exceptions::PyRuntimeError::new_err("iterate the reader before calling next")),
            Ok(None) => Ok(None),
            Ok(Some(Err(e))) => Err(to_py_error(e)),
            Ok(Some(Ok(output))) => Ok(Some(output_to_py(py, output)?)),
        }
    }

    /// The number of shards the configuration expands to.
    fn shard_count(&self) -> PyResult<usize> {
        if self.config.verbatim {
            return Ok(self.config.urls.len());
        }
        let mut total = 0;
        for pattern in &self.config.urls {
            total += webdataset::core::utils::expand_urls(pattern).map_err(to_py_error)?.len();
        }
        Ok(total)
    }

    fn __repr__(&self) -> String {
        format!("<webdataset._native.Reader {} shards>", self.config.urls.len())
    }
}

/// Build the iterator a configuration describes.
fn make_iterator(config: &Config) -> PyResult<Box<dyn Iterator<Item = webdataset::Result<Output>> + Send>> {
    let pipeline = build(config)?;
    let samples = pipeline.iter();

    use webdataset::filters::TupleIteratorExt;

    Ok(match (&config.to_tuple, config.batchsize, config.collate) {
        (Some(specs), None, _) => Box::new(samples.to_tuple(specs.clone()).map(|row| row.map(Output::Row))),
        (Some(specs), Some(size), true) => Box::new(
            TupleIteratorExt::batched(samples.to_tuple(specs.clone()), size, config.partial)
                .map(|row| row.map(Output::CollatedRow)),
        ),
        (Some(specs), Some(size), false) => Box::new(
            TupleIteratorExt::listed(samples.to_tuple(specs.clone()), size, config.partial)
                .map(|rows| rows.map(Output::RowBatch)),
        ),
        (None, Some(size), true) => {
            Box::new(samples.batched(size, config.partial).map(|batch| batch.map(Output::Collated)))
        }
        (None, Some(size), false) => {
            Box::new(samples.listed(size, config.partial).map(|group| group.map(Output::Batch)))
        }
        (None, None, _) => Box::new(samples.map(|sample| sample.map(Output::Sample))),
    })
}

/// Turn one unit of output into the Python object the caller expects.
fn output_to_py(py: Python<'_>, output: Output) -> PyResult<Py<PyAny>> {
    Ok(match output {
        Output::Sample(sample) => sample_to_py(py, &sample)?.into_any().unbind(),
        Output::Row(row) => {
            let items: Vec<Bound<'_, PyAny>> =
                row.iter().map(|value| value_to_py(py, value)).collect::<PyResult<_>>()?;
            PyTuple::new(py, items)?.into_any().unbind()
        }
        Output::Batch(group) => {
            let list = PyList::empty(py);
            for sample in &group {
                list.append(sample_to_py(py, sample)?)?;
            }
            list.into_any().unbind()
        }
        Output::Collated(batch) => sample_to_py(py, &batch)?.into_any().unbind(),
        Output::CollatedRow(columns) => {
            let items: Vec<Bound<'_, PyAny>> =
                columns.iter().map(|value| value_to_py(py, value)).collect::<PyResult<_>>()?;
            PyTuple::new(py, items)?.into_any().unbind()
        }
        Output::RowBatch(rows) => {
            let list = PyList::empty(py);
            for row in &rows {
                let items: Vec<Bound<'_, PyAny>> =
                    row.iter().map(|value| value_to_py(py, value)).collect::<PyResult<_>>()?;
                list.append(PyTuple::new(py, items)?)?;
            }
            list.into_any().unbind()
        }
    })
}

/// Decode one sample, as the pipeline's decode stage would.
///
/// The Python layer calls this when a pipeline could not be lowered whole, so
/// that decoding still has exactly one implementation.
#[pyfunction]
#[pyo3(signature = (sample, spec = "basic", only = None))]
fn decode_sample(
    py: Python<'_>,
    sample: &Bound<'_, PyDict>,
    spec: &str,
    only: Option<Vec<String>>,
) -> PyResult<Py<PyAny>> {
    let mut decoder = match spec {
        "" | "basic" => Decoder::default(),
        _ => Decoder::new(vec![Arc::new(webdataset::ImageHandler::parse(spec).map_err(to_py_error)?)]),
    };
    if let Some(fields) = only {
        decoder = decoder.only(fields);
    }

    let input = convert::py_to_sample(sample)?;
    let decoded = decoder.decode(input).map_err(to_py_error)?;
    Ok(sample_to_py(py, &decoded)?.into_any().unbind())
}

/// Expand a brace expression, as the `braceexpand` package does.
#[pyfunction]
fn braceexpand(pattern: &str) -> PyResult<Vec<String>> {
    webdataset::braceexpand(pattern).map_err(to_py_error)
}

/// Expand a shard specification, including `::` separators.
#[pyfunction]
fn expand_urls(spec: &str) -> PyResult<Vec<String>> {
    webdataset::core::utils::expand_urls(spec).map_err(to_py_error)
}

/// Split a path into its basename and its full extension.
#[pyfunction]
fn base_plus_ext(path: &str) -> Option<(String, String)> {
    webdataset::core::utils::base_plus_ext(path).map(|(a, b)| (a.to_string(), b.to_string()))
}

/// Encode a list of NumPy arrays in the tenbin format.
#[pyfunction]
fn tenbin_encode(py: Python<'_>, arrays: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
    let mut tensors = Vec::with_capacity(arrays.len());
    for array in arrays {
        tensors.push(writer::numpy_to_tensor(&array)?);
    }
    let encoded = webdataset::tenbin::encode_buffer(&tensors);
    Ok(pyo3::types::PyBytes::new(py, &encoded).into_any().unbind())
}

/// Decode a tenbin byte string into a list of NumPy arrays.
#[pyfunction]
fn tenbin_decode(py: Python<'_>, data: &[u8]) -> PyResult<Py<PyAny>> {
    let tensors = webdataset::tenbin::decode_buffer(data).map_err(to_py_error)?;
    let list = PyList::empty(py);
    for tensor in &tensors {
        list.append(convert::tensor_to_py(py, tensor)?)?;
    }
    Ok(list.into_any().unbind())
}

/// Which worker and node this thread is, as the splitters see it.
#[pyfunction]
fn worker_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let info = webdataset::core::worker_info();
    let dict = PyDict::new(py);
    dict.set_item("rank", info.rank)?;
    dict.set_item("world_size", info.world_size)?;
    dict.set_item("worker", info.worker)?;
    dict.set_item("num_workers", info.num_workers)?;
    Ok(dict.into_any().unbind())
}

/// Present this thread as worker `worker` of `num_workers`.
#[pyfunction]
fn set_worker(worker: usize, num_workers: usize) {
    webdataset::core::utils::set_worker(worker, num_workers);
}

/// The version of the Rust implementation underneath.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Whether JPEG is decoded by libjpeg-turbo rather than the pure-Rust decoder.
///
/// With it, JPEG decoding is bit-identical to Pillow's; without, the two differ
/// by a count or two, since the standard does not fix the inverse DCT.
#[pyfunction]
fn has_libjpeg() -> bool {
    cfg!(feature = "libjpeg")
}

/// The native half of the `webdataset` package.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reader>()?;
    m.add_class::<writer::TarWriter>()?;
    m.add_class::<writer::ShardWriter>()?;
    m.add_function(wrap_pyfunction!(decode_sample, m)?)?;
    m.add_function(wrap_pyfunction!(braceexpand, m)?)?;
    m.add_function(wrap_pyfunction!(expand_urls, m)?)?;
    m.add_function(wrap_pyfunction!(base_plus_ext, m)?)?;
    m.add_function(wrap_pyfunction!(tenbin_encode, m)?)?;
    m.add_function(wrap_pyfunction!(tenbin_decode, m)?)?;
    m.add_function(wrap_pyfunction!(worker_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_worker, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(has_libjpeg, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
