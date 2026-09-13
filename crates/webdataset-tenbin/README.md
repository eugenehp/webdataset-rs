# webdataset-tenbin

The tenbin (`.ten`) binary tensor format used by
[WebDataset](https://github.com/webdataset/webdataset).

Tenbin is an 8-byte aligned encoding for lists of dense arrays. Because the
payload is aligned and stored in native byte order, a decoder can hand the
bytes straight to a compute kernel or an RDMA transfer without copying.

```rust
use webdataset_core::Tensor;
use webdataset_tenbin::{decode_buffer, encode_buffer};

let tensors = vec![Tensor::from_f32_shaped(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])];
let encoded = encode_buffer(&tensors);
assert_eq!(decode_buffer(&encoded).unwrap(), tensors);
```

## Format

A file is a sequence of chunks:

```text
magic     8 bytes, "~TenBin~"
length    8 bytes, native-endian i64, the unpadded payload length
payload   `length` bytes, zero padded to a multiple of 64
```

Each array contributes a header chunk and a data chunk. The header is itself a
run of 8-byte fields: the short NumPy dtype name, an 8-byte label, the rank,
and then the shape.

## `no_std`

The in-memory encode and decode functions need only `core` and `alloc`; the
stream and file helpers need `std`.

```toml
webdataset-tenbin = { version = "0.1", default-features = false }
```

License: BSD-3-Clause
