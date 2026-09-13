# webdataset (Python)

A drop-in replacement for the Python
[`webdataset`](https://github.com/webdataset/webdataset) library, backed by
Rust.

Existing code runs unchanged:

```python
import webdataset as wds

dataset = (
    wds.WebDataset(url, shardshuffle=100)
    .shuffle(1000)
    .decode("rgb")
    .to_tuple("png", "cls")
    .batched(64)
)
```

Both interfaces are supported — the fluid one above and the explicit
`DataPipeline(...)` form — along with `TarWriter`, `ShardWriter`, the handlers,
`tenbin`, the shard lists, and the rest of the documented surface.

## Building

```console
$ pip install maturin
$ maturin develop --release
```

`libjpeg-turbo` must be installed and findable by `pkg-config`: JPEG is decoded
through the same library Pillow uses, which is what makes the pixels identical.

## How much runs in Rust

A pipeline made only of stages this library implements natively — shard
listing, archive reading, grouping, decoding, shuffling, batching, field
selection — runs entirely in Rust with the GIL released, and only finished
samples cross back into Python. Adding a stage that runs Python code keeps the
fast path for everything before it and applies that stage as a generator on
top. Ask what a pipeline will do:

```python
>>> ds = wds.WebDataset(url).shuffle(100).decode("rgb8").to_tuple("png", "cls").batched(8)
>>> ds.explain()
{'native': ['read', 'shuffle', 'decode', 'to_tuple', 'batched'], 'python': [], ...}

>>> ds = wds.WebDataset(url).decode().map(my_function).to_tuple("png", "cls")
>>> ds.explain()
{'native': ['read', 'decode'], 'python': ['map', 'to_tuple'], ...}
```

## Is it really a drop-in?

Checked, not asserted. `parity/run_api_parity.py` runs 24 pipelines under both
implementations through the public API and compares every value — 7,825 fields,
all identical, covering every decoder and every `imagespec`. The same
comparison is a test, so a regression fails the suite:

```console
$ WDS_REFERENCE_PYTHON=ref/bin/python pytest crates/webdataset-python/tests -q
55 passed
```

## Speed

Around **2.7x** by the clock over the reference implementation on a
decode-and-batch pipeline, and **7x** where the work is archive parsing rather
than image decoding. See the workspace README for the full table.

License: BSD-3-Clause
