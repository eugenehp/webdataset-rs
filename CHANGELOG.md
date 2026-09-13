# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] — unreleased

First release: a Rust implementation of the
[WebDataset](https://github.com/webdataset/webdataset) format, checked against
the reference Python library field by field.

### Added

**Reading and writing.** A streaming tar parser handling ustar, GNU long names
and PAX extended headers, with compression detected from the shard's leading
bytes rather than its name. Shards read from local disk, a pipe, HTTP, Google
Cloud Storage, AIStore or the Hugging Face Hub; a local shard cache with atomic
downloads, format validation and LRU eviction. `TarWriter` and `ShardWriter`
write shards back out.

**Pipelines.** A re-runnable stage pipeline, so `repeat` and `with_epoch` work.
Shard lists with brace expansion, resampling, and node and worker splitting.
Filters for shuffling, decoding, mapping, selecting, renaming, projecting to
tuples and batching. A threaded loader.

**Decoding.** `.txt`, `.cls`, `.json`, `.npy`, `.npz`, `.ten`, `.cbor`, `.mp`
and images, with gzip chaining. Every `imagespec` the reference library
defines, from `rgb8` through `torchrgba8` and `pil`.

**Robustness against untrusted shards.** The tar and tenbin parsers are fuzzed
against malformed input, including headers that carry a valid checksum over
nonsense fields, since those are what get past a cheap rejection and into the
parsing. A shard is untrusted input — downloaded, or written by someone else —
so a broken one must produce an error the handler can act on rather than taking
the process down. Reading a payload now grows with the bytes that arrive
instead of reserving the size the header declares: a crafted header naming more
memory than the machine has would otherwise abort the process before any error
could be returned. The same applies to tenbin's length-prefixed chunks.

**Async.** The same parser, decoders, shuffling and batching driven by futures
and producing a `Stream`, with configurable shard concurrency so fetches
overlap.

**WebAssembly.** The whole stack builds for `wasm32-unknown-unknown` and
`wasm32-wasip1`. Threads and subprocesses are features rather than assumptions,
and `MemoryOpener` takes shard bytes from the host.

**`no_std`.** `webdataset-core` and `webdataset-tenbin` build against `core`
and `alloc` alone, verified on a bare-metal ARM target. Per-thread worker
identity works without the standard library through a host-supplied thread
token.

**Python bindings.** `webdataset-python` builds a drop-in replacement for the
reference library. A pipeline made only of natively implemented stages runs
entirely in Rust with the GIL released; a Python stage keeps the fast path for
everything before it.

**Tooling.** `wds`, a command line tool for inspecting and reshaping shards.
`parity/`, which compares this implementation against the reference field by
field. `bench/`, which times the two.

### Parity

Checked rather than asserted, and enforced by the test suites:

- 5,646 fields comparing the Rust library against the reference, across every
  bundled shard and every `imagespec`.
- 7,825 fields comparing the Python bindings against the reference through the
  public API, across 24 pipelines.
- JPEG decoding is bit-identical with the `libjpeg` feature, which decodes
  through the same libjpeg-turbo that Pillow uses. Without it the pure-Rust
  decoder differs by at most one count on real photographs, since the JPEG
  standard fixes the transform but not the inverse DCT.

### Differences from the reference implementation

- Errors are values. Rust iterators cannot unwind and resume, so "re-raise"
  means "forward the error downstream" rather than "raise an exception".
- Python pickles (`.pkl`, `.pyd`, `.pth`) are reported as unsupported rather
  than executed.
- Workers are threads rather than processes.
- Transformations that change the item type are iterator adapters rather than
  pipeline stages, because a Rust pipeline is homogeneous.
- A truncated shard cannot corrupt the next one: a shard boundary is emitted
  even when a shard fails part way through.
