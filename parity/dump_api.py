#!/usr/bin/env python3
"""Dump what a pipeline yields, using only the public ``webdataset`` API.

Run this under the reference implementation and under the Rust-backed one and
diff the output: that is the drop-in claim, checked rather than asserted. Values
are reduced to a canonical byte form first, so the comparison is about the data
and not about which object happens to be holding it.
"""

import argparse
import json
import pathlib
import sys

import webdataset as wds

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _canonical import describe, digest


def pipelines(url, image="jpg"):
    """The pipelines to compare, written the way the library documents them.

    Nothing here shuffles: the two implementations draw from different random
    number generators by design, so comparing orders would say nothing about
    whether they read the same data.
    """
    return {
        "raw": lambda: wds.WebDataset(url, shardshuffle=False),
        "decode": lambda: wds.WebDataset(url, shardshuffle=False).decode(),
        "decode_rgb8": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgb8"),
        "decode_rgb": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgb"),
        "decode_l8": lambda: wds.WebDataset(url, shardshuffle=False).decode("l8"),
        "decode_rgba8": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgba8"),
        "decode_torchrgb8": lambda: wds.WebDataset(url, shardshuffle=False).decode("torchrgb8"),
        "to_tuple": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgb8").to_tuple(image, "cls"),
        "batched": lambda: (
            wds.WebDataset(url, shardshuffle=False).decode("rgb8").to_tuple(image, "cls").batched(8)
        ),
        "batched_dict": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgb8").batched(8),
        "listed": lambda: wds.WebDataset(url, shardshuffle=False).decode().listed(8),
        "rename": lambda: wds.WebDataset(url, shardshuffle=False).decode().rename(image=image, label="cls"),
        "select": lambda: (
            wds.WebDataset(url, shardshuffle=False).decode().select(lambda s: int(s["cls"]) % 2 == 0)
        ),
        "map": lambda: (
            wds.WebDataset(url, shardshuffle=False).decode().map(lambda s: {**s, "extra": len(s[image])})
        ),
        "map_dict": lambda: wds.WebDataset(url, shardshuffle=False).decode().map_dict(cls=lambda c: c * 2),
        "map_tuple": lambda: (
            wds.WebDataset(url, shardshuffle=False)
            .decode()
            .to_tuple("cls", image)
            .map_tuple(lambda c: c + 1, None)
        ),
        "explicit": lambda: wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.tarfile_to_samples(),
            wds.decode("rgb8"),
            wds.to_tuple(image, "cls"),
            wds.batched(8),
        ),
        "slice": lambda: wds.WebDataset(url, shardshuffle=False).decode().slice(10),
        "unbatched": lambda: wds.WebDataset(url, shardshuffle=False).decode("rgb8").batched(8).unbatched(),
        "with_epoch": lambda: wds.WebDataset(url, shardshuffle=False).decode().with_epoch(20),
        # Exercises .npy, .npz and .cbor when the corpus was built with --extras.
        "decode_extras": lambda: wds.WebDataset(url, shardshuffle=False).decode(),
        # `select_files` and `rename_files` are applied per archive member,
        # before grouping, so a rename can change which sample a file lands in.
        "select_files": lambda: wds.WebDataset(
            url, shardshuffle=False, select_files=lambda name: name.endswith(".cls")
        ),
        "rename_files": lambda: wds.WebDataset(
            url, shardshuffle=False, rename_files=lambda name: name.replace(".cls", ".label")
        ).decode(),
        # Changes every key, so a rename applied after grouping rather than
        # before would silently leave the keys alone.
        "rename_files_changes_key": lambda: wds.WebDataset(
            url, shardshuffle=False, rename_files=lambda name: "renamed_" + name
        ),
        # Merges pairs of samples, which must raise the same way on both sides
        # once two files land under the same extension.
        "rename_files_regroups": lambda: wds.WebDataset(
            url,
            shardshuffle=False,
            rename_files=lambda name: name.rpartition(".")[0][:-1] + "." + name.rpartition(".")[2],
            select_files=lambda name: name.endswith(".cls"),
        ),
    }


def digest_item(item):
    """Reduce one yielded item to a comparable record."""
    if isinstance(item, dict):
        fields = {
            name: {"type": describe(value), "sha256": digest(value)}
            for name, value in item.items()
            # `__local_path__` records where a shard happened to live, which is
            # not part of the data.
            if name != "__local_path__"
        }
        return {"kind": "dict", "fields": fields}
    if isinstance(item, (list, tuple)):
        return {
            "kind": "sequence",
            "fields": {
                str(i): {"type": describe(value), "sha256": digest(value)} for i, value in enumerate(item)
            },
        }
    return {
        "kind": "scalar",
        "fields": {"0": {"type": describe(item), "sha256": digest(item)}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image", default="jpg", help="the image field name in the corpus")
    parser.add_argument("--list", action="store_true", help="print the pipeline names and exit")
    args = parser.parse_args()

    if args.list:
        for name in pipelines(args.url, args.image):
            print(name)
        return 0

    build = pipelines(args.url, args.image).get(args.pipeline)
    if build is None:
        print(f"unknown pipeline {args.pipeline}", file=sys.stderr)
        return 2

    # A pipeline that raises is still comparable: the two implementations
    # should fail in the same way, at the same point. Only the exception type
    # is compared, since the wording is not part of the contract.
    try:
        for index, item in enumerate(build()):
            if args.limit and index >= args.limit:
                break
            print(json.dumps(digest_item(item), sort_keys=True))
    except Exception as exn:
        print(json.dumps({"kind": "error", "fields": {"type": {"type": type(exn).__name__, "sha256": ""}}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
