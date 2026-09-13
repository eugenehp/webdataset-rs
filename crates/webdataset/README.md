# webdataset

High-performance sequential dataset loading from tar archives, in the
[WebDataset](https://github.com/webdataset/webdataset) format.

A WebDataset is a set of tar files ("shards"). Inside a shard, the files that
share a basename make up one training sample:

```text
imagenet-000000.tar
  n03991062_24866.jpg   n03991062_24866.cls
  n03995372_9042.jpg    n03995372_9042.cls
```

That is the entire format. There is no index, no metadata file and no
conversion step. Because reading is sequential, a shard streams at the full
bandwidth of the device or the network link, and a dataset is just a list of
URLs.

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
    }
    Ok(())
}
```

## What this crate gives you

- A re-runnable pipeline of stages, so `repeat` and `with_epoch` work.
- Decoders for `.txt`, `.cls`, `.json`, `.npy`, `.npz`, `.ten`, `.cbor`,
  `.mp` and images, plus gzip chaining.
- Shard lists with brace expansion, resampling, and node and worker splitting.
- A threaded loader, and an async pipeline over `futures_io::AsyncRead`.
- Writers, from a single archive to a numbered series of shards.

## Features

| feature | adds |
|---|---|
| `threads` *(default)* | per-shard read-ahead and the multi-worker loader |
| `subprocess` *(default)* | `pipe:` and the `curl`/`gsutil`/`ais` schemes |
| `yaml` *(default)* | multi-source dataset specifications |
| `async` | reading from any `AsyncRead`, yielding a `Stream` |
| `image` | `.jpg`, `.png` and friends |
| `libjpeg` | decode JPEG with libjpeg-turbo, matching Pillow bit for bit |
| `msgpack`, `cbor`, `npz` | those formats |
| `zstd`, `bzip2`, `xz` | shards in those containers |
| `wasm-js` | host randomness on `wasm32-unknown-unknown` |
| `full` | every format, plus threads, subprocesses and async |

## Parity with the reference implementation

This is a port, not a binding, and its output is checked against the Python
library field by field — 5,646 fields across every bundled shard and every
`imagespec`, all identical. See the
[workspace README](https://github.com/eugenehp/webdataset-rs) for how that is
measured and what the deliberate differences are.

License: BSD-3-Clause
