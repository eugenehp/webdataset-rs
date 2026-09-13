#!/usr/bin/env python3
"""Dump a WebDataset shard's contents, as the reference Python library sees it.

Every sample becomes one JSON line holding its key, its fields, and a SHA-256
digest of each decoded field. The Rust dumper emits exactly the same shape, so
the two can be compared line by line to check that this port reads the format
identically.

Digests are taken over a canonical byte form of the decoded value, not over the
Python object, so that two implementations agree whenever the *decoded data*
agrees rather than when their in-memory representations happen to match.
"""

import argparse
import json
import pathlib
import sys

import webdataset as wds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _canonical import describe, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", help="shard URL or brace pattern")
    parser.add_argument("--decode", default="none", help="none, basic, or an imagespec such as rgb8")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many samples")
    args = parser.parse_args()

    dataset = wds.DataPipeline(wds.SimpleShardList(args.shards), wds.tarfile_to_samples())
    if args.decode == "basic":
        dataset = dataset.compose(wds.decode())
    elif args.decode != "none":
        dataset = dataset.compose(wds.decode(args.decode))

    for index, sample in enumerate(dataset):
        if args.limit and index >= args.limit:
            break
        fields = {}
        for name, value in sample.items():
            if name.startswith("__"):
                continue
            fields[name] = {"type": describe(value), "sha256": digest(value)}
        print(json.dumps({"key": sample["__key__"], "fields": fields}, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
