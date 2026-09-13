#!/usr/bin/env python3
"""Compare this port against the reference Python implementation.

Both implementations dump every sample of a shard as one JSON line: the sample
key, and for each field a type tag plus a SHA-256 of a canonical byte form of
the decoded value. Two implementations agree on a field when the digests match,
which is a statement about the decoded data rather than about how either one
represents it in memory.

Run with the Python interpreter that has `webdataset` installed:

    parity/run_parity.py --rust ./target/release/parity-dump
"""

import argparse
import json
import pathlib
import subprocess
import sys

# (shard, decode mode) pairs to compare.
CASES = [
    ("testdata/sample.tgz", "none"),
    ("testdata/imagenet-000000.tgz", "none"),
    ("testdata/mpdata.tar", "none"),
    ("testdata/tendata.tar", "none"),
    ("testdata/testgz.tar", "none"),
    ("testdata/compressed.tar", "none"),
    ("testdata/ixtest.tar", "none"),
    ("testdata/sample.tgz", "basic"),
    ("testdata/imagenet-000000.tgz", "basic"),
    ("testdata/mpdata.tar", "basic"),
    ("testdata/tendata.tar", "basic"),
    ("testdata/testgz.tar", "basic"),
    ("testdata/compressed.tar", "basic"),
    ("testdata/imagenet-000000.tgz", "rgb8"),
    ("testdata/imagenet-000000.tgz", "rgb"),
    ("testdata/imagenet-000000.tgz", "l8"),
    ("testdata/imagenet-000000.tgz", "rgba8"),
    ("testdata/imagenet-000000.tgz", "torchrgb8"),
    ("testdata/imagenet-000000.tgz", "torchrgb"),
    ("testdata/imagenet-000000.tgz", "torch"),
    ("testdata/imagenet-000000.tgz", "torchl8"),
    ("testdata/imagenet-000000.tgz", "torchrgba8"),
    ("testdata/imagenet-000000.tgz", "pil"),
    ("testdata/sample.tgz", "rgb8"),
    # Real JPEGs, decoded. Exact only when the `libjpeg` feature is built in;
    # without it the pure-Rust decoder differs by a count or two, which
    # `parity/jpeg_divergence.py` measures separately. These pages are
    # 1938x3002, so a handful is plenty and a full pass is needlessly slow.
    ("testdata/ixtest.tar", "rgb8", 3),
    ("testdata/ixtest.tar", "l8", 3),
    ("testdata/ixtest.tar", "torchrgb8", 3),
    ("testdata/imagenet-000000.tgz::testdata/sample.tgz", "none"),
]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stderr[-2000:]}")
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def compare(python_samples, rust_samples):
    """Count matching fields, and collect the first few differences."""
    total = matched = 0
    problems = []

    if len(python_samples) != len(rust_samples):
        problems.append(f"sample count: python {len(python_samples)}, rust {len(rust_samples)}")

    for index, (want, got) in enumerate(zip(python_samples, rust_samples)):
        if want["key"] != got["key"]:
            problems.append(f"sample {index}: key {want['key']!r} vs {got['key']!r}")
            continue
        names = set(want["fields"]) | set(got["fields"])
        for name in sorted(names):
            total += 1
            a = want["fields"].get(name)
            b = got["fields"].get(name)
            if a is None or b is None:
                problems.append(f"{want['key']}.{name}: present in {'python' if a else 'rust'} only")
            elif a["sha256"] != b["sha256"]:
                problems.append(f"{want['key']}.{name}: value differs ({a['type']} vs {b['type']})")
            elif a["type"] != b["type"]:
                problems.append(f"{want['key']}.{name}: type {a['type']} vs {b['type']} (values agree)")
                matched += 1
            else:
                matched += 1
    return total, matched, problems


def check(rust_dumper, cases=None, limit=0, python=None):
    """Compare every case and return a structured result.

    Returns ``(total, matched, per_case)`` where ``per_case`` maps
    ``(shard, decode)`` to ``(total, matched, problems)``. Tests should assert
    on it; :func:`main` prints a table and sets the exit status.
    """
    here = pathlib.Path(__file__).resolve().parent
    dumper = str(here / "dump_python.py")
    interpreter = python or sys.executable

    grand_total = grand_matched = 0
    per_case = {}

    for case in cases or CASES:
        shard, decode = case[0], case[1]
        # A case may carry its own limit, for shards whose images are large.
        case_limit = case[2] if len(case) > 2 else limit
        common = ["--decode", decode] + (["--limit", str(case_limit)] if case_limit else [])
        try:
            python_samples = run([interpreter, dumper, shard, *common])
            rust_samples = run([rust_dumper, shard, *common])
        except RuntimeError as exn:
            per_case[(shard, decode)] = (0, 0, [str(exn)])
            continue
        total, matched, problems = compare(python_samples, rust_samples)
        grand_total += total
        grand_matched += matched
        per_case[(shard, decode)] = (total, matched, problems)

    return grand_total, grand_matched, per_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", default="./target/release/parity-dump")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    dumper = str(here / "dump_python.py")

    grand_total = grand_matched = 0
    failures = []

    print(f"{'shard':<40} {'decode':<10} {'fields':>8} {'match':>8}  status")
    print("-" * 82)

    for case in CASES:
        shard, decode = case[0], case[1]
        case_limit = case[2] if len(case) > 2 else args.limit
        common = ["--decode", decode]
        if case_limit:
            common += ["--limit", str(case_limit)]
        try:
            python_samples = run([sys.executable, dumper, shard, *common])
            rust_samples = run([args.rust, shard, *common])
        except RuntimeError as exn:
            failures.append((shard, decode, [str(exn)]))
            print(f"{pathlib.Path(shard).name:<40} {decode:<10} {'-':>8} {'-':>8}  ERROR")
            continue

        total, matched, problems = compare(python_samples, rust_samples)
        grand_total += total
        grand_matched += matched
        share = 100.0 * matched / total if total else 100.0
        status = "ok" if matched == total and not problems else "MISMATCH"
        if status != "ok":
            failures.append((shard, decode, problems))
        name = shard if len(shard) <= 40 else "..." + shard[-37:]
        print(f"{name:<40} {decode:<10} {total:>8} {share:>7.1f}%  {status}")

    print("-" * 82)
    share = 100.0 * grand_matched / grand_total if grand_total else 100.0
    print(f"{'TOTAL':<40} {'':<10} {grand_total:>8} {share:>7.1f}%")

    if failures:
        print("\ndifferences:")
        for shard, decode, problems in failures:
            print(f"\n  {shard} [{decode}]")
            shown = problems if args.verbose else problems[:5]
            for problem in shown:
                print(f"    {problem}")
            if len(problems) > len(shown):
                print(f"    ... and {len(problems) - len(shown)} more")

    return 0 if grand_matched == grand_total and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
