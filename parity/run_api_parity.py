#!/usr/bin/env python3
"""Check that the Rust-backed package is a drop-in for the reference one.

Both interpreters run ``parity/dump_api.py``, which exercises a pipeline
through the public API and prints a digest per yielded item. The two outputs
must agree line for line.

    parity/run_api_parity.py \\
        --reference /path/to/reference/bin/python \\
        --rust /path/to/rust/bin/python \\
        --url 'corpus/bench-{000000..000003}.tar'
"""

import argparse
import json
import pathlib
import subprocess
import sys

PIPELINES = [
    "raw",
    "decode",
    "decode_rgb8",
    "decode_rgb",
    "decode_l8",
    "decode_rgba8",
    "to_tuple",
    "batched",
    "batched_dict",
    "listed",
    "rename",
    "select",
    "map",
    "map_dict",
    "map_tuple",
    "explicit",
    "slice",
    "unbatched",
    "with_epoch",
    "decode_extras",
    "select_files",
    "rename_files",
    "rename_files_changes_key",
    "rename_files_regroups",
]


def run(interpreter, dumper, url, pipeline, limit, image):
    command = [interpreter, dumper, url, "--pipeline", pipeline, "--image", image]
    if limit:
        command += ["--limit", str(limit)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr else "failed")
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def compare(want, got):
    """Count matching fields and collect the first differences."""
    total = matched = 0
    problems = []

    if len(want) != len(got):
        problems.append(f"item count: reference {len(want)}, rust {len(got)}")

    for index, (a, b) in enumerate(zip(want, got)):
        if a["kind"] != b["kind"]:
            problems.append(f"item {index}: kind {a['kind']} vs {b['kind']}")
            continue
        # Both raised: the exception type is the thing to agree on.
        if a["kind"] == "error":
            total += 1
            x, y = a["fields"]["type"]["type"], b["fields"]["type"]["type"]
            if x == y:
                matched += 1
            else:
                problems.append(f"item {index}: raised {x} vs {y}")
            continue
        names = set(a["fields"]) | set(b["fields"])
        for name in sorted(names):
            total += 1
            x, y = a["fields"].get(name), b["fields"].get(name)
            if x is None or y is None:
                problems.append(f"item {index}, field {name}: only in {'reference' if x else 'rust'}")
            elif x["sha256"] != y["sha256"]:
                problems.append(f"item {index}, field {name}: value differs ({x['type']} vs {y['type']})")
            elif x["type"] != y["type"]:
                problems.append(f"item {index}, field {name}: type {x['type']} vs {y['type']} (values agree)")
                matched += 1
            else:
                matched += 1
    return total, matched, problems


def check(reference, rust, url, *, image="png", limit=0, pipelines=None):
    """Compare every pipeline and return a structured result.

    Returns ``(total, matched, per_pipeline)`` where ``per_pipeline`` maps a
    name to ``(total, matched, problems)``. Callers that want a hard failure
    should assert on it; :func:`main` prints a table and sets the exit status.
    """
    dumper = str(pathlib.Path(__file__).resolve().parent / "dump_api.py")
    grand_total = grand_matched = 0
    per_pipeline = {}

    for pipeline in pipelines or PIPELINES:
        try:
            want = run(reference, dumper, url, pipeline, limit, image)
            got = run(rust, dumper, url, pipeline, limit, image)
        except RuntimeError as exn:
            per_pipeline[pipeline] = (0, 0, [str(exn)])
            continue
        total, matched, problems = compare(want, got)
        grand_total += total
        grand_matched += matched
        per_pipeline[pipeline] = (total, matched, problems)

    return grand_total, grand_matched, per_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="interpreter with the reference webdataset")
    parser.add_argument("--rust", required=True, help="interpreter with the Rust-backed webdataset")
    parser.add_argument("--url", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--image", default="png", help="the image field name in the corpus")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    grand_total, grand_matched, per_pipeline = check(
        args.reference, args.rust, args.url, image=args.image, limit=args.limit
    )
    failures = []

    print(f"{'pipeline':<20} {'fields':>8} {'match':>8}  status")
    print("-" * 50)

    for pipeline, (total, matched, problems) in per_pipeline.items():
        if total == 0 and problems:
            failures.append((pipeline, problems))
            print(f"{pipeline:<20} {'-':>8} {'-':>8}  ERROR")
            continue
        share = 100.0 * matched / total if total else 100.0
        status = "ok" if matched == total and not problems else "MISMATCH"
        if status != "ok":
            failures.append((pipeline, problems))
        print(f"{pipeline:<20} {total:>8} {share:>7.1f}%  {status}")

    print("-" * 50)
    share = 100.0 * grand_matched / grand_total if grand_total else 100.0
    print(f"{'TOTAL':<20} {grand_total:>8} {share:>7.1f}%")

    if failures:
        print("\ndifferences:")
        for pipeline, problems in failures:
            print(f"\n  {pipeline}")
            shown = problems if args.verbose else problems[:5]
            for problem in shown:
                print(f"    {problem}")
            if len(problems) > len(shown):
                print(f"    ... and {len(problems) - len(shown)} more")

    return 0 if grand_matched == grand_total and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
