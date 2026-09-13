# Releasing

## Before

```console
$ cargo test --workspace --all-features
$ cargo clippy --workspace --all-features --all-targets
$ cargo fmt --all --check
$ RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
$ ruff check . && ruff format --check .
```

Then the parity runs, which are what the release actually rests on:

```console
$ cargo build --release -p webdataset-tools --features libjpeg
$ python parity/run_parity.py                         # Rust vs the reference
$ python parity/run_api_parity.py --reference ... --rust ...   # the bindings
$ python parity/jpeg_divergence.py --reference ... --rust ...
```

All three must report 100%. The same checks run as assertions under
`pytest crates/webdataset-python/tests`, with `WDS_REFERENCE_PYTHON` set.

## Version

Versions are inherited from `[workspace.package]` in the root `Cargo.toml`;
bump it there and every crate follows. Update `CHANGELOG.md` in the same
commit, moving the `unreleased` heading to the new version and its date.

## Rust crates

```console
$ scripts/publish.sh              # check everything, upload nothing
$ scripts/publish.sh --execute    # upload
```

The script runs the tests, lints and docs first, checks that the version has
not drifted between the workspace, the path-dependency pins and the bindings
crate, then uploads in dependency order.

It publishes one crate at a time and skips versions already on crates.io.
`cargo publish --workspace` is all-or-nothing: when one crate is rejected the
command aborts and whatever it already sent stays sent. That happened here —
`webdataset-tar` turned out to be taken, and the run stopped with three of six
crates published and no way to repeat it cleanly. Crate by crate, a rejection
stops at that crate, and the next run carries on.

Verification is the other way round: it runs for the workspace at once, because
a crate cannot be checked on its own until everything it depends on is on the
index.

`webdataset-tools` and `webdataset-python` are `publish = false` — the first
holds the parity dumper and the README compile check, the second goes to PyPI.

### If verification fails on a crate you have not changed

Cargo keeps extracted copies of locally packaged crates under
`~/.cargo/registry/src/`, and does not always refresh them when a crate is
renamed or a version is reused. The symptom is a build error naming a module
that no longer exists. Delete the directory holding them — it contains nothing
but copies of this workspace, and cargo rebuilds it:

```console
$ ls ~/.cargo/registry/src/          # the one that is not index.crates.io-*
$ rm -rf ~/.cargo/registry/src/<that-one>
```

## Python package

```console
$ cd crates/webdataset-python
$ maturin build --release --out dist
```

`--auditwheel=repair` copies libjpeg-turbo and liblzma into the wheel, so it
does not depend on the build machine's Homebrew or apt tree. Check the wheel in
a clean environment before uploading:

```console
$ python -m venv /tmp/check && /tmp/check/bin/pip install dist/*.whl pytest
$ /tmp/check/bin/python -c "import webdataset; print(webdataset._native.has_libjpeg())"
```

That must print `True`; a wheel built without libjpeg-turbo would decode JPEG
with the pure-Rust decoder and no longer match Pillow exactly.

Then:

```console
$ maturin upload dist/*
```

Release wheels should be built under `manylinux` for Linux and for each macOS
architecture; `maturin build` in a container or on CI handles that.

The wheel links libjpeg-turbo and liblzma, which are system libraries. An
unrepaired wheel records absolute paths to them and fails to import on any
machine that does not have them at exactly those paths. `auditwheel = "repair"`
in `pyproject.toml` copies them into the wheel and rewrites the references, so a
plain `maturin build` is enough; check the result with

```console
$ otool -L webdataset/_native.abi3.so     # macOS
$ ldd webdataset/_native.abi3.so          # Linux
```

Nothing outside `/usr/lib` should remain.
