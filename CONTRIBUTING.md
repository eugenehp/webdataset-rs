# Contributing

## Getting set up

```console
$ cargo test --workspace
```

That is enough for most changes.

The `libjpeg` feature links libjpeg-turbo 3.x. It uses the system copy when
`pkg-config` finds one new enough, and builds one from source otherwise — which
is what happens on most distributions, since they lag well behind 3.0. The
source build wants `nasm` on x86: without it libjpeg-turbo silently drops its
SIMD paths, and the decoder is then not the one the JPEG parity numbers were
measured against. On ARM the SIMD is plain C intrinsics and needs nothing. A few parts need more:

| to work on | you also need |
|---|---|
| the `libjpeg` feature | `cmake`, and `nasm` on x86 |
| the Python bindings | `maturin`, plus the above |
| the parity checks | a second interpreter, see below |
| WebAssembly | `rustup target add wasm32-unknown-unknown wasm32-wasip1` |
| `no_std` | `rustup target add thumbv7em-none-eabi` |

## Before opening a pull request

```console
$ cargo test --workspace --all-features
$ cargo clippy --workspace --all-features --all-targets
$ cargo fmt --all --check
$ ruff check . && ruff format --check .
```

CI runs those plus the reduced feature sets, the cross-compilation targets, and
the parity comparison.

## What the parity checks are for

This library is a port of the Python `webdataset`, and its value rests on
behaving identically. That is checked rather than assumed: `parity/` runs the
same pipelines under both implementations and compares every decoded value.

The comparison needs an interpreter with the reference library and everything
it decodes with. Leave any of these out and cases are skipped rather than run,
which reads as a pass:

```console
$ python -m venv /tmp/reference
$ /tmp/reference/bin/pip install webdataset numpy pillow msgpack cbor torch
```

`torch` is needed for the `torch*` imagespecs and `cbor` for building the test
corpus. Then export `WDS_REFERENCE_PYTHON=/tmp/reference/bin/python`.

If you change anything a sample passes through — the archive parser, a decoder,
the grouping rule, a filter — run the comparison:

```console
$ cargo build --release -p webdataset-tools --features libjpeg
$ python parity/run_parity.py
```

It must report 100%. The same checks run as assertions under `pytest
crates/webdataset-python/tests` when `WDS_REFERENCE_PYTHON` is set.

Deliberate differences from the reference are listed at the end of the README.
If you are adding one, add it there too, with the reason.

## Tests

Unit tests live beside the code; integration tests are in each crate's
`tests/`. A test should say what behaviour it pins down, not restate the code:
`reads_the_same_samples_from_every_shard` rather than `test_read_2`.

Two habits worth keeping:

- **Check that a new assertion can fail.** Break the thing on purpose and
  confirm the test notices. An assertion that cannot fail is worse than none,
  because it reads as evidence. `test_the_comparison_can_actually_fail` exists
  for exactly this reason.
- **Prefer a fixture over a mock** where the format is involved. The shards in
  `testdata/` come from the reference implementation, so reading them correctly
  means something.

## Benchmarks

```console
$ cargo bench -p webdataset
```

Build their shards in memory rather than reading `testdata/`, so the numbers
measure this library rather than the disk. Take them on an idle machine; a
busy one makes wall-clock times meaningless, which is why `bench/compare.py`
reports CPU time alongside.

## Style

The minimum supported Rust version is 1.85, and the `msrv` CI job checks it
against every feature. If a dependency bump raises it, hold the dependency back
rather than raising the floor — that is a decision for a release, not for a
dependency update. The reason is recorded next to each pinned version.

`rustfmt` and `ruff format` settle formatting. Beyond that: comments should say
why, not what, and public items need a doc comment — `missing_docs` is denied
across the workspace.
