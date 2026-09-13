#!/usr/bin/env python3
"""Time the reference implementation against the Rust-backed one.

Both interpreters run ``bench/benchmark.py``, which builds each pipeline with
the public API, so the user-facing code is identical and only what is
underneath differs.

    bench/compare.py \\
        --reference /path/to/reference/bin/python \\
        --rust /path/to/rust/bin/python \\
        --url 'corpus/bench-{000000..000015}.tar'
"""

import argparse
import json
import pathlib
import subprocess
import sys


def run(interpreter, script, url, repeats):
    command = [interpreter, script, url, "--repeats", str(repeats), "--json"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-2000:])
    return json.loads(result.stdout)


def rate(seconds, items):
    return items / seconds if seconds > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rust", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=0, help="samples per pass, for the rate column")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="note in the output that the machine was busy, so wall times should be discounted",
    )
    args = parser.parse_args()

    script = str(pathlib.Path(__file__).resolve().parent / "benchmark.py")
    reference = run(args.reference, script, args.url, args.repeats)
    rust = run(args.rust, script, args.url, args.repeats)

    print(f"reference  {reference['path']}  ({reference['implementation']})")
    print(f"rust       {rust['path']}  ({rust['implementation']})")
    print()
    print(f"each figure is the best of {args.repeats} runs")
    if args.cpu:
        print("wall-clock times are unreliable on a busy machine; CPU time is the one to trust")
    print()

    header = f"{'pipeline':<20} {'ref wall':>10} {'rust wall':>10} {'x':>6}"
    header += f"   {'ref cpu':>10} {'rust cpu':>10} {'x':>6}"
    if args.samples:
        header += f" {'samples/s':>11}"
    print(header)
    print("-" * len(header))

    wall_speedups, cpu_speedups = [], []
    for name, want in reference["results"].items():
        got = rust["results"].get(name, {})
        if "error" in want or "error" in got:
            print(f"{name:<20} {want.get('error') or got.get('error')}")
            continue

        wall = want["best"] / got["best"] if got["best"] > 0 else float("inf")
        cpu = want["cpu"] / got["cpu"] if got["cpu"] > 0 else float("inf")
        wall_speedups.append(wall)
        cpu_speedups.append(cpu)

        line = f"{name:<20} {want['best']:9.3f}s {got['best']:9.3f}s {wall:5.1f}x"
        line += f"   {want['cpu']:9.3f}s {got['cpu']:9.3f}s {cpu:5.1f}x"
        if args.samples:
            line += f" {rate(got['cpu'], args.samples):10.0f}"
        print(line)

    print("-" * len(header))
    if cpu_speedups:
        wall_speedups.sort()
        cpu_speedups.sort()
        wall_median = wall_speedups[len(wall_speedups) // 2]
        cpu_median = cpu_speedups[len(cpu_speedups) // 2]
        blank = f"{'':>10} {'':>10}"
        print(f"{'median speedup':<20} {blank} {wall_median:5.1f}x   {blank} {cpu_median:5.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
