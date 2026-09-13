# webdataset-core

Core data model for the [WebDataset](https://github.com/webdataset/webdataset)
format: `Sample`, `Value`, `Tensor`, the `.npy` codec, brace expansion, error
types and pluggable handlers.

Most users want the [`webdataset`](https://crates.io/crates/webdataset) crate,
which re-exports everything here. Depend on this one directly if you only need
the data model — to write a tool that manipulates samples, or to build for a
target the full stack does not reach.

## `no_std`

Builds against `core` and `alloc` alone:

```toml
webdataset-core = { version = "0.1", default-features = false, features = ["json"] }
```

`std` adds the parts that need an operating system: `Error::Io`, environment
variable substitution, and the stream-based `npy` helpers. The `threads`
feature keeps per-worker state per thread rather than per program; without the
standard library there is no portable way to ask which thread is running, so
the host supplies one with `workers::set_thread_id_hook`. A single-threaded
`no_std` program needs neither.

## Features

| feature | adds |
|---|---|
| `std` *(default)* | filesystem errors, environment variables, stream helpers |
| `json` *(default)* | conversion between `Value` and `serde_json::Value` |
| `threads` *(default)* | per-thread worker identity |
| `image` | an `Image` variant on `Value` |

License: BSD-3-Clause
