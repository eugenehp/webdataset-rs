#!/usr/bin/env python3
"""Time a set of pipelines against whichever ``webdataset`` is installed.

Run it once under each interpreter and compare; ``bench/compare.py`` does that
and prints the table. Every pipeline here is written the way the reference
library documents, so the two runs execute identical user code.
"""

import argparse
import json
import statistics
import sys
import time

import webdataset as wds


def timed(build, repeats, warmup=1):
    """Return per-run wall and CPU times, after discarding warm-up runs.

    CPU time is reported alongside wall time because it is what survives a busy
    machine: it counts only this process's work, across all its threads, so a
    competing build changes it far less than it changes the clock on the wall.
    """
    wall, cpu, counted = [], [], 0
    for run in range(repeats + warmup):
        started, started_cpu = time.perf_counter(), time.process_time()
        count = 0
        for _ in build():
            count += 1
        elapsed, elapsed_cpu = time.perf_counter() - started, time.process_time() - started_cpu
        if run >= warmup:
            wall.append(elapsed)
            cpu.append(elapsed_cpu)
            counted = count
    return wall, cpu, counted


def pipelines(url):
    """The pipelines to time, each as a name and a builder."""

    def raw():
        return wds.WebDataset(url, shardshuffle=False)

    def decode_basic():
        return wds.WebDataset(url, shardshuffle=False).decode()

    def decode_images():
        return wds.WebDataset(url, shardshuffle=False).decode("rgb8")

    def to_tuple():
        return wds.WebDataset(url, shardshuffle=False).decode("rgb8").to_tuple("jpg", "cls")

    def batched():
        return wds.WebDataset(url, shardshuffle=False).decode("rgb8").to_tuple("jpg", "cls").batched(64)

    def shuffled():
        return (
            wds.WebDataset(url, shardshuffle=100)
            .shuffle(1000)
            .decode("rgb8")
            .to_tuple("jpg", "cls")
            .batched(64)
        )

    def explicit():
        return wds.DataPipeline(
            wds.SimpleShardList(url),
            wds.tarfile_to_samples(),
            wds.shuffle(1000),
            wds.decode("rgb8"),
            wds.to_tuple("jpg", "cls"),
            wds.batched(64),
        )

    def with_python_map():
        # A Python stage in the middle: the point is to show what it costs.
        return wds.WebDataset(url, shardshuffle=False).decode("rgb8").map(lambda s: s).to_tuple("jpg", "cls")

    return [
        ("read only", raw),
        ("decode basic", decode_basic),
        ("decode rgb8", decode_images),
        ("+ to_tuple", to_tuple),
        ("+ batched(64)", batched),
        ("+ shuffle(1000)", shuffled),
        ("explicit pipeline", explicit),
        ("with a python map", with_python_map),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="shard URL or brace pattern")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=0, help="samples per shard set, for the rate")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = {}
    for name, build in pipelines(args.url):
        try:
            wall, cpu, count = timed(build, args.repeats)
        except Exception as exn:
            results[name] = {"error": f"{type(exn).__name__}: {exn}"}
            continue
        results[name] = {
            "best": min(wall),
            "median": statistics.median(wall),
            "cpu": min(cpu),
            "cpu_median": statistics.median(cpu),
            "count": count,
            "samples": args.samples or count,
        }

    report = {"implementation": wds.__version__, "path": wds.__file__, "results": results}
    if args.json:
        print(json.dumps(report))
    else:
        print(f"{wds.__file__}  (version {wds.__version__})")
        for name, r in results.items():
            if "error" in r:
                print(f"  {name:<20} {r['error']}")
            else:
                print(f"  {name:<20} wall {r['best']:7.3f}s  cpu {r['cpu']:7.3f}s  ({r['count']} items)")


if __name__ == "__main__":
    sys.exit(main())
