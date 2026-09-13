"""Assert that this package is a drop-in for the reference implementation.

``test_api.py`` checks the shape of what the package returns. This checks it
against the thing it claims to replace: the same pipelines run under both
implementations, and every value compared.

That needs two interpreters, since only one ``webdataset`` can be importable at
a time. Point ``WDS_REFERENCE_PYTHON`` at an interpreter with the reference
library installed and these run; without it they skip, so the suite still works
for anyone who just wants to test this package:

    python -m venv /tmp/reference
    /tmp/reference/bin/pip install webdataset numpy pillow msgpack cbor torch
    WDS_REFERENCE_PYTHON=/tmp/reference/bin/python pytest crates/webdataset-python/tests
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "parity"))

import jpeg_divergence  # noqa: E402
import run_api_parity  # noqa: E402
import run_parity  # noqa: E402
import webdataset  # noqa: E402

REFERENCE = os.environ.get("WDS_REFERENCE_PYTHON")

needs_reference = pytest.mark.skipif(
    not REFERENCE,
    reason="set WDS_REFERENCE_PYTHON to an interpreter with the reference webdataset installed",
)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """A lossless corpus covering every decoder.

    PNG rather than JPEG on purpose: PNG is lossless, so any difference in the
    decoded pixels is a real defect rather than the inverse-DCT slack the JPEG
    standard permits. ``--extras`` adds ``.npy``, ``.npz`` and ``.cbor``, so no
    decoder goes unexercised.
    """
    directory = tmp_path_factory.mktemp("corpus")
    result = subprocess.run(
        [
            REFERENCE or sys.executable,
            str(ROOT / "bench" / "make_corpus.py"),
            str(directory),
            "--shards",
            "2",
            "--per-shard",
            "32",
            "--format",
            "png",
            "--extras",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the corpus: {result.stderr.strip()[-400:]}")
    return str(directory / "bench-{000000..000001}.tar")


@needs_reference
def test_every_pipeline_is_identical(corpus):
    """Every value, from every pipeline, must match the reference exactly."""
    total, matched, per_pipeline = run_api_parity.check(REFERENCE, sys.executable, corpus, image="png")

    assert total > 0, "the comparison ran no pipelines, so it proved nothing"

    # Name the offending pipeline and field, rather than just the totals: a
    # bare count tells you something broke but not where to look.
    differing = {
        name: problems[:3]
        for name, (fields, agreed, problems) in per_pipeline.items()
        if problems or agreed != fields
    }
    assert not differing, f"pipelines that differ from the reference: {differing}"
    assert matched == total, f"only {matched} of {total} fields matched"


#: With libjpeg-turbo, JPEG decoding is the same library Pillow uses and the
#: two agree exactly. Without it the pure-Rust decoder is still correct, but the
#: standard does not fix the inverse DCT, so a count or two of slack remains.
EXACT_JPEG = webdataset._native.has_libjpeg()


@needs_reference
def test_jpeg_decoding_matches_the_reference():
    """Real photographs, decoded both ways."""
    stats = jpeg_divergence.measure(
        REFERENCE, sys.executable, str(ROOT / "testdata" / "ixtest.tar"), limit=10
    )

    assert stats["images"] >= 5, "too few images compared for the result to mean anything"
    if EXACT_JPEG:
        assert stats["largest"] == 0, f"libjpeg-turbo should match Pillow exactly, got {stats['largest']}"
        assert stats["identical"] == 1.0
    else:
        assert stats["largest"] <= 1, f"expected at most one count of slack, got {stats['largest']}"
        assert stats["within_one"] == 1.0


@needs_reference
def test_jpeg_decoding_matches_on_pathological_input(tmp_path):
    """Random noise is the worst case for a JPEG decoder; check it too."""
    result = subprocess.run(
        [
            REFERENCE,
            str(ROOT / "bench" / "make_corpus.py"),
            str(tmp_path),
            "--shards",
            "1",
            "--per-shard",
            "32",
            "--format",
            "jpeg",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"could not build the corpus: {result.stderr.strip()[-400:]}")

    stats = jpeg_divergence.measure(REFERENCE, sys.executable, str(tmp_path / "bench-000000.tar"), limit=32)

    if EXACT_JPEG:
        assert stats["largest"] == 0, f"libjpeg-turbo should match Pillow exactly, got {stats['largest']}"
    else:
        assert stats["largest"] <= 4, f"worst-case divergence {stats['largest']} exceeds the bound of 4"
        assert stats["within_two"] > 0.99, f"only {100 * stats['within_two']:.2f}% of pixels within +/-2"


@needs_reference
def test_the_rust_library_matches_the_reference(tmp_path):
    """The same check one level down: the Rust library against the reference.

    ``test_every_pipeline_is_identical`` compares the two *Python* packages,
    which shares the Rust reader with this one but goes through the bindings.
    This compares the Rust library directly, over the bundled shards and every
    ``imagespec``, so a defect in the reader is caught even if the bindings
    happen to paper over it.
    """
    dumper = ROOT / "target" / "release" / "parity-dump"
    if not dumper.exists():
        pytest.skip("build it first: cargo build --release -p webdataset-tools --features libjpeg")

    # A cheap but complete case list: every decoder and every imagespec, but
    # against the small shards. The full sweep, which also runs the 13 MB
    # ImageNet shard through each imagespec, is `parity/run_parity.py`, and CI
    # runs that separately. A test needs to catch regressions quickly.
    small = "testdata/sample.tgz"
    cases = [
        (small, "none"),
        (small, "basic"),
        ("testdata/tendata.tar", "basic"),
        ("testdata/mpdata.tar", "basic"),
        ("testdata/compressed.tar", "basic"),
        ("testdata/testgz.tar", "basic"),
        ("testdata/ixtest.tar", "none"),
    ] + [(small, spec) for spec in ["rgb8", "rgb", "l8", "rgba8", "torchrgb8", "torchrgb", "pil"]]

    total, matched, per_case = run_parity.check(str(dumper), cases=cases, limit=24, python=REFERENCE)

    assert total > 0, "the comparison ran no cases, so it proved nothing"
    differing = {
        f"{pathlib.Path(shard).name} [{decode}]": problems[:3]
        for (shard, decode), (fields, agreed, problems) in per_case.items()
        if problems or agreed != fields
    }
    assert not differing, f"cases that differ from the reference: {differing}"
    assert matched == total, f"only {matched} of {total} fields matched"


@needs_reference
def test_the_comparison_can_actually_fail(corpus, monkeypatch):
    """Guard against a check that always passes.

    An assertion that cannot fail is worse than none, because it reads as
    evidence. This feeds the comparison a deliberately corrupted digest and
    requires it to notice.
    """
    real_compare = run_api_parity.compare

    def corrupted(want, got):
        if got and got[0]["fields"]:
            first = next(iter(got[0]["fields"]))
            got[0]["fields"][first] = dict(got[0]["fields"][first], sha256="0" * 64)
        return real_compare(want, got)

    monkeypatch.setattr(run_api_parity, "compare", corrupted)
    total, matched, per_pipeline = run_api_parity.check(
        REFERENCE, sys.executable, corpus, image="png", pipelines=["raw"]
    )

    assert matched < total, "the comparison did not notice a corrupted value"
    assert per_pipeline["raw"][2], "the comparison reported no problem for a corrupted value"
