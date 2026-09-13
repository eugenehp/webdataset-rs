# webdataset-shard

Streaming tar readers and writers for the
[WebDataset](https://github.com/webdataset/webdataset) format.

A shard is an ordinary tar archive whose members are named `<key>.<extension>`;
the members sharing a key make up one sample. This crate reads shards into
samples and writes them back out, in both cases as a stream — the archive is
never seeked, so the same code works over HTTP, a pipe, or local disk.

```rust
use webdataset_shard::{group_by_keys, open_shard};

for sample in group_by_keys(open_shard("shard-000000.tar")?) {
    let sample = sample?;
    println!("{} has {:?}", sample.key().unwrap_or(""), sample.field_names());
}
# Ok::<(), webdataset_core::Error>(())
```

The tar parser is written here rather than borrowed, so that it can be driven
both synchronously and from a future, and so that shard boundaries and
malformed archives behave predictably. It handles ustar, GNU long names and PAX
extended headers, and is checked against the `tar` crate byte for byte.

Compression is detected from the shard's leading bytes rather than its name, so
`.tar`, `.tar.gz` and `.tgz` all just work.

## Features

| feature | adds |
|---|---|
| `threads` *(default)* | per-shard read-ahead on a background thread |
| `subprocess` *(default)* | the non-local URL schemes; see `webdataset-io` |
| `async` | reading from any `futures_io::AsyncRead`, yielding a `Stream` |
| `zstd`, `bzip2`, `xz` | shards in those containers |

License: BSD-3-Clause
