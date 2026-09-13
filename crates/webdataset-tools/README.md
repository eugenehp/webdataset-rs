# webdataset-tools

Repository tooling, not published.

| binary | does |
|---|---|
| `parity-dump` | writes the digest stream `parity/` compares against the reference implementation |
| `readme-snippets` | fails the build if the README's Rust examples stop compiling |

Neither belongs in `webdataset`'s own `examples/`, where they would ship to
anyone depending on the crate.

```console
$ cargo build --release -p webdataset-tools --features libjpeg
$ ./target/release/parity-dump testdata/sample.tgz --decode rgb8
```
