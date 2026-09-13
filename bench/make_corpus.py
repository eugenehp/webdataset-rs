#!/usr/bin/env python3
"""Build a synthetic corpus for benchmarking.

Written with the reference implementation's own writer so that neither side is
reading data shaped to suit it. Images are a fixed size, so that batching can
collate them, and vary in content so that JPEG sizes are realistic.
"""

import argparse
import io
import json
import pathlib
import sys

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="directory to write shards into")
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--per-shard", type=int, default=512)
    parser.add_argument("--size", type=int, default=64, help="image edge, in pixels")
    parser.add_argument(
        "--extras",
        action="store_true",
        help="also write .npy, .npz and .cbor fields, so every decoder is exercised",
    )
    parser.add_argument(
        "--format",
        default="jpeg",
        choices=["jpeg", "png"],
        help="JPEG is realistic for benchmarking; PNG is lossless, so decoders must agree exactly",
    )
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    total = 0
    for shard in range(args.shards):
        path = out / f"bench-{shard:06d}.tar"
        with open(path, "wb") as raw:
            import tarfile

            with tarfile.open(fileobj=raw, mode="w|") as tar:

                def add(name, payload):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(payload))

                for i in range(args.per_shard):
                    key = f"sample{shard:04d}{i:06d}"
                    pixels = rng.integers(0, 256, (args.size, args.size, 3), dtype=np.uint8)
                    buffer = io.BytesIO()
                    if args.format == "jpeg":
                        Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
                    else:
                        Image.fromarray(pixels).save(buffer, format="PNG")

                    add(f"{key}.{'jpg' if args.format == 'jpeg' else 'png'}", buffer.getvalue())
                    add(f"{key}.cls", str(i % 1000).encode())
                    add(f"{key}.json", json.dumps({"index": i, "shard": shard}).encode())

                    if args.extras:
                        vector = np.arange(6, dtype=np.float32) * (i + 1)
                        matrix = rng.standard_normal((3, 4)).astype(np.float64)

                        npy = io.BytesIO()
                        np.lib.format.write_array(npy, vector)
                        add(f"{key}.npy", npy.getvalue())

                        npz = io.BytesIO()
                        np.savez(npz, vector=vector, matrix=matrix)
                        add(f"{key}.npz", npz.getvalue())

                        import cbor

                        add(f"{key}.cbor", cbor.dumps({"index": i, "name": f"item {i}"}))
                    total += 1
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    size = sum(p.stat().st_size for p in out.glob("bench-*.tar"))
    print(f"\n{total} samples in {args.shards} shards, {size / 1e6:.1f} MB total", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
