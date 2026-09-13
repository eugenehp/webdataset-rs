# webdataset-cli

`wds`, a command line tool for inspecting and reshaping
[WebDataset](https://github.com/webdataset/webdataset) shards.

```console
$ cargo install webdataset-cli
```

Shards are ordinary tar files, so `tar` already works on them. What `tar` does
not know is that files sharing a basename are one sample, which is what every
subcommand here is built around.

```console
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
```

| command | does |
|---|---|
| `wds ls` | list the samples in a set of shards |
| `wds info` | summarise samples, fields and sizes |
| `wds cat` | copy samples from several shards into one archive |
| `wds split` | rewrite samples into a new series of shards |
| `wds extract` | write each sample out as files in a directory |
| `wds create` | build shards from a directory of files |

License: BSD-3-Clause
