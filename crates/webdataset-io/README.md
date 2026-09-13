# webdataset-io

URL opening and shard caching for the
[WebDataset](https://github.com/webdataset/webdataset) format.

`gopen` turns any shard URL into a byte stream:

| scheme | handled by |
|---|---|
| *(none)*, `file:` | the local filesystem |
| `pipe:` | the system shell |
| `http:`, `https:`, `ftp:`, `ftps:`, `sftp:`, `scp:` | `curl` |
| `gs:` | `gsutil` |
| `htgs:` | `curl`, against `storage.googleapis.com` |
| `ais:` | `ais` |
| `hf:` | `curl`, against `huggingface.co` |

Shelling out keeps credentials and proxy configuration in the transfer tools,
exactly as the Python implementation does. Add your own schemes with
`register_scheme`.

`FileCache` layers a local shard cache on top, with atomic downloads, format
validation and LRU eviction.

## Features

| feature | adds |
|---|---|
| `subprocess` *(default)* | every scheme above except the local filesystem |

Turning `subprocess` off leaves the local filesystem and whatever schemes you
register, which is the shape this crate takes on WebAssembly.

License: BSD-3-Clause
