# webdataset-rs

A Rust implementation of [WebDataset](https://github.com/webdataset/webdataset): high-performance,
purely sequential dataset loading from tar archives.

A WebDataset is a set of tar files ("shards"). Inside a shard, the files that share a basename
make up one training sample:

```
imagenet-000000.tar
  n03991062_24866.jpg   n03991062_24866.cls
  n03995372_9042.jpg    n03995372_9042.cls
  ...
```

That is the entire format. There is no index, no metadata file, and no conversion step — any
collection of tar files following the convention is a dataset, and standard tools still work on it.
Because reading is sequential, a shard streams at the full bandwidth of the disk or the network
link, which is typically several times faster than random access over many small files, and it
works the same way over HTTP or against an object store.

## Contents

- [Quick start](#quick-start) — read a dataset, write a dataset
- [The prelude](#the-prelude) — the one-line import
- [The workspace](#the-workspace) — what each crate is for
- [How a pipeline fits together](#how-a-pipeline-fits-together)
- [URLs](#urls) — schemes, brace expansion
- [Scaling out](#scaling-out) — workers, nodes, resampling
- [Errors](#errors) — handlers
- [Async](#async) · [WebAssembly](#webassembly) · [`no_std`](#no_std)
- [Python bindings](#python-bindings) — the drop-in replacement, and its speed
- [Verifying parity](#verifying-parity) — how the claims are checked
- [Differences from the Python implementation](#differences-from-the-python-implementation)

## Quick start

```toml
[dependencies]
webdataset = "0.1"
```

```rust
use webdataset::filters::{SampleIteratorExt, TupleIteratorExt};
use webdataset::{Decoder, Result, WebDataset};

fn main() -> Result<()> {
    let dataset = WebDataset::builder("https://host/imagenet-{000000..000146}.tar")
        .shard_shuffle(100)
        .cache_dir("./_cache")
        .build()?
        .shuffle(1000)
        .decode(Decoder::default());

    for batch in dataset.iter().to_tuple(["jpg;png", "cls"]).batched(64, true) {
        let batch = batch?;
        let (images, labels) = (&batch[0], &batch[1]);
        // train on the batch
    }
    Ok(())
}
```

Writing is symmetric:

```rust
use std::sync::Arc;
use webdataset::{DefaultEncoder, Result, Sample, ShardWriter, Value};

fn main() -> Result<()> {
    let mut writer = ShardWriter::new("train-%06d.tar")?
        .with_encoder(Arc::new(DefaultEncoder::new()))
        .with_max_count(10_000);

    let mut sample = Sample::with_key("sample000001");
    sample.insert("cls", Value::Int(7));
    sample.insert("txt", Value::Text("a caption".into()));
    writer.write(&sample)?;
    writer.close()
}
```

## The prelude

Most pipelines need the same handful of types, plus two extension traits whose
methods are invisible unless the trait is in scope:

```rust
use webdataset::prelude::*;
```

That carries `WebDataset`, `Sample`, `Value`, `Tensor`, the error and handler
types, `Selection`, the writers, and the `SampleIteratorExt` /
`TupleIteratorExt` traits that supply `.shuffled()`, `.select()`, `.decode()`
and the rest. Anything more specialised — the shard list implementations, the
mixers, the encoder — still comes from its own module.

`examples/prelude_check.rs` is built by `cargo test`, so if the prelude ever
stops being enough for an ordinary pipeline, the suite says so.

## Requirements

Rust 1.85 or later, which is what edition 2024 needs. That is checked on every
change, across the full feature set, so it is a promise rather than an
aspiration; a dependency that would raise it is held back instead.

Optional features pull in more: `libjpeg` needs libjpeg-turbo 3.x, using the
system copy if `pkg-config` finds one and building it from source otherwise
(`cmake`, plus `nasm` on x86); `subprocess` needs a shell. Neither is on by
default.

## The workspace

| crate | what it does |
|---|---|
| [`webdataset`](crates/webdataset) | the library you depend on: pipelines, filters, decoding, the loader |
| [`webdataset-core`](crates/webdataset-core) | `Sample`, `Value`, `Tensor`, errors, handlers, brace expansion |
| [`webdataset-shard`](crates/webdataset-shard) | streaming tar readers and writers |
| [`webdataset-io`](crates/webdataset-io) | URL opening (`gopen`) and the local shard cache |
| [`webdataset-tenbin`](crates/webdataset-tenbin) | the `.ten` 8-byte-aligned tensor format |
| [`webdataset-cli`](crates/webdataset-cli) | the `wds` command line tool |
| [`webdataset-python`](crates/webdataset-python) | Python bindings: a drop-in replacement for the reference library |

Alongside them: [`parity/`](parity) compares this implementation against the
reference field by field, [`bench/`](bench) times the two,
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers working on it, and
[`RELEASING.md`](RELEASING.md) describes how a release is cut.

For the Rust code on its own there are criterion benchmarks covering archive
parsing, each decoder, shuffling, collation and a whole pipeline:

```console
$ cargo bench -p webdataset
$ cargo bench -p webdataset -- decoders     # just the field decoders
```

They build their shards in memory, so they measure the library rather than the
disk and need no fixtures.

The lower crates are usable on their own — `webdataset-shard` to walk a shard, `webdataset-tenbin`
to read a `.ten` file — but most users want the top-level `webdataset` crate, which re-exports
everything.

## How a pipeline fits together

`WebDataset` assembles the standard stages, but each one is public and a pipeline can be built by
hand:

```
SimpleShardList      a list of shard URLs
  -> SplitByNode     keep this distributed rank's shards
  -> SplitByWorker   keep this loader worker's shards
  -> Shuffle         shuffle the shard order
  -> ShardsToSamples open each shard, group its files into samples
  -> Shuffle         shuffle samples within a buffer
  -> Decode          turn bytes into images, tensors, JSON
```

Everything above produces `Sample`s, so the whole pipeline can be re-run each epoch — that is what
`repeat` and `with_epoch` need. Transformations that change the item type, such as projecting to
tuples or batching, are ordinary iterator adapters applied to `dataset.iter()`.

## URLs

`gopen` dispatches on the URL scheme, shelling out to the usual transfer tools so that credentials
and proxy settings stay where you already configured them:

| scheme | handled by |
|---|---|
| *(none)*, `file:` | the local filesystem |
| `pipe:` | the system shell |
| `http:`, `https:`, `ftp:`, `ftps:`, `sftp:`, `scp:` | `curl` |
| `gs:` | `gsutil` |
| `htgs:` | `curl`, against `storage.googleapis.com` |
| `ais:` | `ais` |
| `hf:` | `curl`, against `huggingface.co` |

Register your own with `webdataset::io::register_scheme`.

Shard lists use brace expansion (`data-{000000..000146}.tar`) and `::` to concatenate sources.

## Scaling out

`DataLoader` runs one copy of the pipeline per worker thread. Shards are divided between workers by
the `SplitByWorker` stage and between distributed processes by `SplitByNode`, both of which read
`RANK`/`WORLD_SIZE` from the environment.

When exact partitioning is awkward — many nodes, few shards — use `.resampled(true)` instead. Each
worker then draws shards with replacement, so no worker runs dry and the epoch length is whatever
you ask for with `with_epoch`.

A pipeline with neither a node splitter nor resampling refuses to start on more than one node,
rather than silently training every rank on the same data.

## Errors

Every stream yields `Result<Sample>`. What happens when a shard is truncated or a field fails to
decode is decided by a `Handler`:

```rust
let dataset = WebDataset::builder("shards-{000..999}.tar")
    .handler(webdataset::handlers::warn_and_continue())
    .build()?;
```

`reraise_exception` (the default) forwards the error to you, `ignore_and_continue` drops the
sample, `warn_and_stop` ends the stream. Streaming a petabyte means meeting some bad bytes, so
`warn_and_continue` is a common choice for training.

## Secure mode

Setting `WDS_SECURE=1`, or calling `webdataset::core::utils::set_enforce_security(true)`, disables
the `pipe:` and `file:` schemes and URL rewriting from the environment. Python pickles are never
decoded, in any mode.

## The `wds` command line tool

```console
$ cargo install webdataset-cli

$ wds info 'data-{000000..000009}.tar'
samples   93122
bytes     11983224714 (12.0 GB)
shards    10
mean size 128.7 kB

field                 count      bytes  present
cls                   93122     272 kB  100.0%
jpg                   93122    12.0 GB  100.0%

$ wds ls data-000000.tar -n 3
n03991062_24866	cls jpg
n03995372_9042	cls jpg
n04004767_3346	cls jpg

$ wds split 'data-{000000..000009}.tar' -o reshuffled-%06d.tar --shuffle 10000 --max-count 5000
$ wds extract data-000000.tar -o ./files -n 100
$ wds create ./files -o rebuilt-%06d.tar
```

## Features

| feature | adds |
|---|---|
| `threads` *(default)* | per-shard read-ahead and the multi-worker loader |
| `subprocess` *(default)* | `pipe:` and the `curl`/`gsutil`/`ais` backed schemes |
| `yaml` *(default)* | multi-source dataset specifications |
| `async` | read shards from any `AsyncRead`, yielding a `Stream` |
| `image` | `.jpg`, `.png` and friends, via the `image` crate |
| `libjpeg` | decode JPEG with libjpeg-turbo, matching Pillow bit for bit |
| `msgpack` | `.mp` and `.msg` |
| `cbor` | `.cbor` |
| `npz` | NumPy `.npz` archives |
| `zstd`, `bzip2`, `xz` | shards in those containers |
| `wasm-js` | draw randomness from the JavaScript host on `wasm32-unknown-unknown` |
| `full` | every format, plus threads, subprocesses and async |

`.npy`, `.ten`, `.json`, `.txt`, `.cls` and gzip need no features.

## Async

The `async` feature mirrors the whole library on futures: same archive parser,
same decoders, same shuffling and batching, yielding a `Stream` instead of an
`Iterator`. Use it when shards arrive over a network — a blocking reader holds a
thread for the length of a transfer, an async one holds only a task.

```rust
use futures_util::TryStreamExt;
use webdataset::asynch::{AsyncSampleStreamExt, AsyncTupleStreamExt, AsyncWebDataset};

let dataset = AsyncWebDataset::builder("https://host/imagenet-{000000..000146}.tar")
    .opener(Arc::new(MyHttpOpener::new()))
    .concurrency(8)          // keep eight shards in flight
    .shard_shuffle(100)
    .build()?
    .shuffle(1000)
    .decode_basic();

let mut batches = dataset.stream().to_tuple(["jpg;png", "cls"]).batched(64, true);
while let Some(batch) = batches.try_next().await? {
    // train on the batch
}
```

Any transport works — implement `AsyncOpener`, which is one method:

```rust
impl AsyncOpener for MyHttpOpener {
    fn open<'a>(&'a self, url: &'a str) -> BoxFuture<'a, Result<AsyncShardSource>> {
        Box::pin(async move {
            let body = self.client.get(url).send().await?.bytes_stream();
            Ok(AsyncShardSource {
                url: url.to_string(),
                local_path: None,
                // Any `Stream` of byte chunks becomes a shard source.
                stream: Box::new(body.map_err(std::io::Error::other).into_async_read()),
            })
        })
    }
}
```

`MemoryOpener` and `FileOpener` are built in. Raising `concurrency` above one
overlaps fetches, which is the point — with four shards behind 250 ms of
latency, `cargo run --example async_pipeline` reports:

```
concurrency 1: 9 batches, 274 samples in 1.09s
concurrency 4: 9 batches, 274 samples in 315.82ms
```

Only the I/O is async. Decoding a JPEG and stacking a batch are computations, so
they are the same code the blocking pipeline runs — which is why the two read
identically, shard for shard and field for field, as `tests/async_equivalence.rs`
asserts. For CPU-bound work that needs parallelism, map it onto your runtime's
blocking pool before it reaches the stream.

## WebAssembly

WebAssembly has no threads to spawn and no processes to run, so both are
features rather than assumptions. Turn them off and supply the shard bytes
yourself:

```toml
[dependencies]
webdataset = { version = "0.1", default-features = false, features = ["image", "wasm-js"] }
```

```rust
use std::sync::Arc;
use webdataset::pipeline::DataPipeline;
use webdataset::shardlists::SimpleShardList;
use webdataset::sources::{MemoryOpener, ShardsToSamples};
use webdataset::stages::{Decode, Shuffle};

// `bytes` came from `fetch()`, an IndexedDB blob, or wherever the host keeps it.
fn dataset(bytes: Vec<u8>) -> DataPipeline {
    let opener = MemoryOpener::new().with("mem://shard-000.tar", bytes);
    DataPipeline::new()
        .with(SimpleShardList::verbatim(["mem://shard-000.tar"]))
        .with(ShardsToSamples::new(Arc::new(opener)))
        .with(Shuffle::new(1000))
        .with(Decode::basic())
}
```

Everything above the transport is unchanged: the same archive parser, the same
decoders, the same shuffling and batching. `wasm32-wasip1` additionally has a
filesystem, so shard paths work there without `MemoryOpener`.

For transports other than memory, register a scheme with
`webdataset::io::register_scheme` and the rest of the pipeline will use it.

## `no_std`

`webdataset-core` and `webdataset-tenbin` build against `core` and `alloc`
alone, which covers the whole data model — samples, values, tensors, the `.npy`
and `.ten` codecs, brace expansion, errors and handlers:

```toml
[dependencies]
webdataset-core = { version = "0.1", default-features = false, features = ["json", "threads"] }
webdataset-tenbin = { version = "0.1", default-features = false }
```

What `std` adds is the parts that need an operating system: `Error::Io`,
environment variables, and the stream and file helpers.

The `threads` feature keeps per-worker state per thread rather than per program.
With `std` that is a thread-local and needs no setup. Without it there is no
portable way to ask which thread is running, so the host says once:

```rust
// On an RTOS this would return the current task's id.
webdataset_core::set_thread_id_hook(|| current_task_id())?;
```

A single-threaded `no_std` program needs neither the feature nor the hook.

The upper crates need `std`: reading a shard means reading a stream, and writing
one means writing a file or a socket.

## Differences from the Python implementation

This is a port, not a binding, and it follows the reference implementation's behaviour closely —
the integration tests assert the same sample counts, image shapes and decoded values as the Python
test suite does against the same shards. A few things necessarily differ:

- **Errors are values.** Rust iterators cannot unwind and resume, so "re-raise" means "forward the
  error downstream" rather than "raise an exception". Handlers otherwise behave the same.
- **No pickles.** `.pkl`, `.pyd` and `.pth` are reported as unsupported instead of being executed.
  Store tensors as `.npy` or `.ten` and structures as `.json`.
- **Workers are threads, not processes**, so there is no fork-safety story to worry about and no
  serialisation between worker and consumer. There is also an async pipeline, which the Python
  implementation has no counterpart to.
- **Type-changing operations are iterator adapters**, not pipeline stages: `to_tuple` and `batched`
  live on `dataset.iter()` rather than on the pipeline, because a Rust pipeline is homogeneous.
- **A truncated shard cannot corrupt the next one.** A shard boundary is emitted even when a shard
  fails part way through, so its trailing files can never merge into the following shard's first
  sample.

## Python bindings

`crates/webdataset-python` builds a Python package that is a drop-in replacement
for the reference `webdataset` library. Existing code runs unchanged:

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

Build it with [maturin](https://www.maturin.rs):

```console
$ cd crates/webdataset-python && maturin develop --release
```

### How much runs in Rust

A pipeline made only of stages this library implements natively — shard listing,
archive reading, grouping, decoding, shuffling, batching, field selection — runs
entirely in Rust with the GIL released, and only finished samples cross back
into Python. Adding a stage that runs Python code keeps the fast path for
everything before it and applies that stage as an ordinary generator on top. Ask
what a pipeline will do:

```python
>>> ds = wds.WebDataset(url).shuffle(100).decode("rgb8").to_tuple("png", "cls").batched(8)
>>> ds.explain()["native"]
['read', 'shuffle', 'decode', 'to_tuple', 'batched']
>>> ds.explain()["python"]
[]

>>> ds = wds.WebDataset(url).decode().map(my_function).to_tuple("png", "cls")
>>> ds.explain()
{'native': ['read', 'decode'], 'python': ['map', 'to_tuple'], ...}
```

### Is it really a drop-in?

`parity/run_api_parity.py` runs the same pipelines under both implementations
through the public API and compares every value. The same comparison is also a
test, so a regression fails the suite rather than waiting for someone to run a
script:

```console
$ WDS_REFERENCE_PYTHON=ref/bin/python pytest crates/webdataset-python/tests -q
51 passed
```

Without `WDS_REFERENCE_PYTHON` the parity tests skip and the rest still run, so
the suite works for anyone who has not set up a second environment. One of
those tests deliberately corrupts a value and requires the comparison to notice
it — an assertion that cannot fail is worse than none, since it reads as
evidence.

Run as a script, it prints the table:

```console
$ parity/run_api_parity.py --reference ref/bin/python --rust rs/bin/python --url 'corpus/{000000..000001}.tar'

pipeline               items   fields    match  status
------------------------------------------------------------
raw                      128      640   100.0%  ok
decode                   128      640   100.0%  ok
decode_rgb8              128      640   100.0%  ok
...
------------------------------------------------------------
TOTAL                            7825   100.0%
```

24 pipelines, 7,825 fields, all identical — covering raw reads, every
`imagespec`, `to_tuple`, batching, collation, `rename`, `select`, `map`,
`map_dict`, `map_tuple`, the explicit pipeline API, slicing, unbatching,
`with_epoch`, and `select_files`/`rename_files`. Build the corpus with
`--extras` and it also covers every decoder: `.json`, `.npy`, `.npz`, `.cbor`,
`.ten` and gzip-chained fields.

Failures are compared too, not just successes. A pipeline that raises must raise
the same kind of exception on both sides, at the same point — that is how a
drop-in behaves when the data is wrong, and it is where the subtler differences
turn up.

**Including JPEG, with the `libjpeg` feature.** The JPEG standard specifies the
transform but not an exact inverse DCT, so two correct decoders can differ in
the last bit — and the pure-Rust one does, by a count or two. Building with
`libjpeg` decodes through libjpeg-turbo, the same library Pillow uses, and the
two then agree exactly. `parity/jpeg_divergence.py` measures it either way:

| build | corpus | identical | largest difference |
|---|---|---|---|
| `libjpeg` | real photographs | **100%** | **0** |
| `libjpeg` | synthetic noise | **100%** | **0** |
| pure Rust | real photographs | 99.75% | 1 |
| pure Rust | synthetic noise *(worst case)* | 72.29% | 4 |

The feature is off by default because it brings a C dependency: libjpeg-turbo
must be installed and findable by `pkg-config`, and it cannot be used on
WebAssembly. The Python bindings turn it on, since matching Pillow is the whole
point of a drop-in replacement.

### Speed

`bench/compare.py` times both implementations over the same corpus, with the
pipelines written identically:

```console
$ bench/compare.py --reference ref/bin/python --rust rs/bin/python       --url 'corpus/bench-{000000..000015}.tar' --repeats 5 --samples 8192
```

8,192 samples of 64×64 JPEGs across 16 shards, 55 MB, best of 5 runs on an idle
machine. Both wall-clock and CPU time are shown, because the gap between them is
itself informative.

| pipeline | ref wall | rust wall | | ref cpu | rust cpu | | samples/s |
|---|---|---|---|---|---|---|---|
| read only | 0.363s | 0.052s | **6.9x** | 0.361s | 0.078s | **4.6x** | 104,900 |
| decode basic | 0.382s | 0.057s | **6.7x** | 0.378s | 0.089s | **4.3x** | 92,400 |
| decode rgb8 | 0.959s | 0.354s | **2.7x** | 0.956s | 0.429s | **2.2x** | 19,100 |
| + to_tuple | 0.968s | 0.362s | **2.7x** | 0.964s | 0.439s | **2.2x** | 18,600 |
| + batched(64) | 1.022s | 0.359s | **2.8x** | 1.020s | 0.434s | **2.4x** | 18,900 |
| + shuffle(1000) | 0.993s | 0.386s | **2.6x** | 0.987s | 0.455s | **2.2x** | 18,000 |
| explicit pipeline | 1.025s | 0.390s | **2.6x** | 1.023s | 0.462s | **2.2x** | 17,700 |
| with a python map | 0.962s | 0.368s | **2.6x** | 0.955s | 0.444s | **2.2x** | 18,500 |

Median **2.7x** by the clock, **2.2x** by CPU, and around **7x** where the work
is archive parsing rather than images.

Those two columns differ for a reason worth knowing. For the reference,
wall-clock and CPU time are the same to three figures — it is one thread doing
one thing. For this library they are not: reading only takes 0.052s of wall
time but 0.078s of CPU, because each shard is parsed on a background thread
while the consumer works. You get the CPU-time speedup in throughput per core,
and the wall-clock one in latency.

The image pipelines gain least, and for a good reason: JPEG decoding dominates
them and is already native code on both sides, so what is left to win is the tar
parsing, the grouping and the per-sample Python object churn.

The last row is the one worth dwelling on. Dropping a Python `.map()` into the
middle of a pipeline still leaves it 1.7x faster, because everything before the
map keeps running in Rust.

## Verifying parity

`parity/` compares this port against the reference implementation directly. Both dump every sample
as one JSON line — the key, and per field a type tag plus a SHA-256 of a canonical byte form of the
decoded value — so a diff is a statement about decoded data rather than about either one's in-memory
representation.

```console
$ pip install webdataset numpy pillow msgpack torch
$ cargo build --release -p webdataset-tools --features libjpeg
$ python parity/run_parity.py

shard                                    decode       fields    match  status
----------------------------------------------------------------------------------
testdata/sample.tgz                      none            180   100.0%  ok
testdata/imagenet-000000.tgz             basic           188   100.0%  ok
testdata/imagenet-000000.tgz             rgb8            188   100.0%  ok
...
----------------------------------------------------------------------------------
TOTAL                                                   5610   100.0%
```

At the time of writing that is 5,610 fields across 25 shard and decode combinations — every raw
byte string, integer, string, JSON tree, tensor and image — all identical, including every
`imagespec` from `rgb8` through `torchrgba8` and `pil`.

## Licence and attribution

BSD 3-Clause, the same as the reference implementation, and for the same
copyright holder: this is a port of the Python
[`webdataset`](https://github.com/webdataset/webdataset) library, Copyright
2020 NVIDIA CORPORATION. The format, the pipeline design, the decoder
behaviour and the API shape all come from it, and much of the behaviour is
reproduced deliberately field for field.

See [LICENSE](LICENSE) for the terms and [NOTICE](NOTICE) for what is derived
from where. The shard fixtures under `testdata/` are copied unchanged from that
project and are used to check that this implementation reads them identically.
