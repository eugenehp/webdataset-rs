#!/usr/bin/env python3
"""Measure how far the two JPEG decoders disagree, and hold them to a bound.

Everything else the two implementations do is bit-identical, which
``run_api_parity.py`` checks against a lossless corpus. JPEG is the exception,
and unavoidably so: the standard specifies the *transform*, not an exact
inverse DCT, so libjpeg (behind PIL) and zune-jpeg (behind the Rust build) are
both correct and still differ in the last bit or two.

Rather than wave that away, this measures it and fails if it exceeds a bound.

    parity/jpeg_divergence.py \\
        --reference /path/to/reference/bin/python \\
        --rust /path/to/rust/bin/python \\
        --url testdata/ixtest.tar

:func:`measure` is the same check as a function, for tests to assert on.
"""

import argparse
import pathlib
import pickle
import subprocess
import sys
import tempfile

#: Decoded under each interpreter in turn, then compared here.
DUMPER = """
import pickle, sys, webdataset as wds
url, out, limit, field = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
images = {}
for i, sample in enumerate(wds.WebDataset(url, shardshuffle=False).decode("rgb8")):
    if i >= limit:
        break
    if field in sample:
        images[sample["__key__"]] = sample[field]
pickle.dump(images, open(out, "wb"))
"""


def decode_with(interpreter, url, limit, field, into, tag):
    """Decode images under one interpreter and bring them back."""
    script = into / "dump.py"
    script.write_text(DUMPER)
    out = into / f"{tag}.pkl"
    result = subprocess.run(
        [interpreter, str(script), url, str(out), str(limit), field],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    with open(out, "rb") as stream:
        return pickle.load(stream)


def measure(reference, rust, url, *, field="jpg", limit=20):
    """Decode the same JPEGs under both and report how far they diverge.

    Returns a dict of counts, the largest per-channel difference, and the PSNR.
    Callers that want a hard failure should assert on ``largest``.
    """
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        into = pathlib.Path(directory)
        want = decode_with(reference, url, limit, field, into, "reference")
        got = decode_with(rust, url, limit, field, into, "rust")

    shared = sorted(set(want) & set(got))
    if not shared:
        raise RuntimeError(f"no `{field}` images found in {url}")
    if set(want) != set(got):
        raise RuntimeError(f"the two runs saw different samples: {len(want)} vs {len(got)}")

    differences, squared = [], []
    for key in shared:
        a, b = want[key].astype(np.int32), got[key].astype(np.int32)
        if a.shape != b.shape:
            raise RuntimeError(f"{key}: shape {a.shape} vs {b.shape}")
        differences.append(np.abs(a - b).ravel())
        squared.append(((a - b) ** 2).mean())

    every = np.concatenate(differences)
    mse = float(np.mean(squared))
    return {
        "images": len(shared),
        "pixels": int(every.size),
        "identical": float((every == 0).mean()),
        "within_one": float((every <= 1).mean()),
        "within_two": float((every <= 2).mean()),
        "largest": int(every.max()),
        "mean": float(every.mean()),
        "psnr": 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rust", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--field", default="jpg")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--max-difference",
        type=int,
        default=4,
        help="the largest per-channel difference to tolerate",
    )
    args = parser.parse_args()

    try:
        stats = measure(args.reference, args.rust, args.url, field=args.field, limit=args.limit)
    except RuntimeError as exn:
        print(exn, file=sys.stderr)
        return 1

    print(f"images compared      {stats['images']}")
    print(f"pixels compared      {stats['pixels']}")
    print(f"identical            {100 * stats['identical']:.2f}%")
    print(f"within +/-1          {100 * stats['within_one']:.2f}%")
    print(f"within +/-2          {100 * stats['within_two']:.2f}%")
    print(f"largest difference   {stats['largest']}")
    print(f"mean difference      {stats['mean']:.4f}")
    print(f"PSNR                 {stats['psnr']:.1f} dB")

    if stats["largest"] > args.max_difference:
        print(f"\nFAIL: largest difference {stats['largest']} exceeds the bound of {args.max_difference}")
        return 1
    print(f"\nok: within the bound of {args.max_difference}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
