"""Tests for the drop-in Python API.

These check the shape of what the package hands back — types, keys, batching,
lowering — rather than comparing against the reference implementation, which is
what ``parity/run_api_parity.py`` is for.

    pytest crates/webdataset-python/tests
"""

import pathlib

import numpy as np
import pytest
import webdataset as wds

TESTDATA = pathlib.Path(__file__).resolve().parents[3] / "testdata"


@pytest.fixture(scope="session")
def imagenet():
    """A bundled shard of PNGs with class labels."""
    return str(TESTDATA / "imagenet-000000.tgz")


@pytest.fixture(scope="session")
def sample_shard():
    """A bundled shard with 90 small samples."""
    return str(TESTDATA / "sample.tgz")


@pytest.fixture
def written(tmp_path):
    """A shard written by this package, for round-trip tests."""
    path = tmp_path / "written.tar"
    with wds.TarWriter(str(path)) as writer:
        for i in range(20):
            writer.write(
                {
                    "__key__": f"sample{i:04d}",
                    "cls": i % 5,
                    "txt": f"sample number {i}",
                    "json": {"index": i, "even": i % 2 == 0},
                    "npy": np.arange(4, dtype=np.float32) * i,
                }
            )
    return str(path)


# -- reading ---------------------------------------------------------------


def test_reads_every_sample(imagenet):
    samples = list(wds.WebDataset(imagenet, shardshuffle=False))
    assert len(samples) == 47
    assert samples[0]["__key__"] == "10"
    assert isinstance(samples[0]["png"], bytes)


def test_undecoded_fields_are_bytes(sample_shard):
    sample = next(iter(wds.WebDataset(sample_shard, shardshuffle=False)))
    assert isinstance(sample["cls"], bytes)
    assert isinstance(sample["png"], bytes)


def test_decode_produces_the_documented_types(imagenet):
    sample = next(iter(wds.WebDataset(imagenet, shardshuffle=False).decode()))
    assert isinstance(sample["cls"], int)
    assert isinstance(sample["png"], bytes), "no imagespec means images stay raw"


def test_decode_imagespec_produces_arrays(imagenet):
    sample = next(iter(wds.WebDataset(imagenet, shardshuffle=False).decode("rgb8")))
    image = sample["png"]
    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.shape == (793, 600, 3)


@pytest.mark.parametrize(
    "spec,dtype,shape",
    [
        ("rgb8", np.uint8, (793, 600, 3)),
        ("rgb", np.float32, (793, 600, 3)),
        ("rgba8", np.uint8, (793, 600, 4)),
        ("l8", np.uint8, (793, 600)),
        ("torchrgb8", np.uint8, (3, 793, 600)),
    ],
)
def test_every_imagespec(imagenet, spec, dtype, shape):
    sample = next(iter(wds.WebDataset(imagenet, shardshuffle=False).decode(spec)))
    assert sample["png"].dtype == dtype
    assert sample["png"].shape == shape


def test_float_images_are_scaled_to_unit_range(imagenet):
    sample = next(iter(wds.WebDataset(imagenet, shardshuffle=False).decode("rgb")))
    assert 0.0 <= sample["png"].min() <= sample["png"].max() <= 1.0


# -- the fluid interface ---------------------------------------------------


def test_to_tuple_projects(imagenet):
    rows = list(wds.WebDataset(imagenet, shardshuffle=False).decode().to_tuple("png", "cls"))
    assert len(rows) == 47
    assert isinstance(rows[0], tuple) and len(rows[0]) == 2
    assert isinstance(rows[0][1], int)


def test_to_tuple_accepts_alternatives(imagenet):
    rows = list(wds.WebDataset(imagenet, shardshuffle=False).decode().to_tuple("jpg;png", "cls"))
    assert len(rows) == 47


def test_batched_collates_columns(imagenet):
    batches = list(wds.WebDataset(imagenet, shardshuffle=False).decode().to_tuple("cls").batched(16))
    assert len(batches) == 3
    assert isinstance(batches[0][0], np.ndarray)
    assert batches[0][0].shape == (16,)
    assert batches[2][0].shape == (15,), "the last batch is partial"


def test_batched_without_partial_drops_the_remainder(imagenet):
    batches = list(
        wds.WebDataset(imagenet, shardshuffle=False).decode().to_tuple("cls").batched(16, partial=False)
    )
    assert len(batches) == 2


def test_listed_does_not_collate(imagenet):
    groups = list(wds.WebDataset(imagenet, shardshuffle=False).listed(16))
    assert len(groups) == 3
    assert isinstance(groups[0], list) and isinstance(groups[0][0], dict)


def test_batched_dict(imagenet):
    batch = next(iter(wds.WebDataset(imagenet, shardshuffle=False).decode().batched(8)))
    assert isinstance(batch, dict)
    assert batch["cls"].shape == (8,)
    assert len(batch["__key__"]) == 8


def test_unbatched_reverses_batching(imagenet):
    samples = list(wds.WebDataset(imagenet, shardshuffle=False).decode().batched(8).unbatched())
    assert len(samples) == 47


def test_shuffle_keeps_every_sample(imagenet):
    keys = [s["__key__"] for s in wds.WebDataset(imagenet, shardshuffle=False).shuffle(20)]
    assert sorted(keys) == sorted(s["__key__"] for s in wds.WebDataset(imagenet, shardshuffle=False))


def test_select_filters(imagenet):
    samples = list(wds.WebDataset(imagenet, shardshuffle=False).decode().select(lambda s: s["cls"] < 100))
    assert 0 < len(samples) < 47
    assert all(s["cls"] < 100 for s in samples)


def test_map_runs_python(imagenet):
    samples = list(
        wds.WebDataset(imagenet, shardshuffle=False).decode().map(lambda s: {**s, "extra": s["cls"] * 2})
    )
    assert samples[0]["extra"] == samples[0]["cls"] * 2


def test_map_dict_transforms_one_field(imagenet):
    samples = list(wds.WebDataset(imagenet, shardshuffle=False).decode().map_dict(cls=lambda c: c + 1000))
    assert all(s["cls"] >= 1000 for s in samples)


def test_rename(imagenet):
    sample = next(
        iter(wds.WebDataset(imagenet, shardshuffle=False).decode().rename(image="png", label="cls"))
    )
    assert "image" in sample and "label" in sample
    assert "png" not in sample


def test_slice_truncates(imagenet):
    assert len(list(wds.WebDataset(imagenet, shardshuffle=False).slice(5))) == 5


def test_with_epoch_repeats(imagenet):
    assert len(list(wds.WebDataset(imagenet, shardshuffle=False).with_epoch(100))) == 100


def test_select_files_drops_members(imagenet):
    samples = list(wds.WebDataset(imagenet, shardshuffle=False, select_files=lambda n: n.endswith(".cls")))
    assert len(samples) == 47
    assert all(sorted(k for k in s if not k.startswith("__")) == ["cls"] for s in samples)


def test_rename_files_rewrites_extensions(imagenet):
    sample = next(
        iter(wds.WebDataset(imagenet, shardshuffle=False, rename_files=lambda n: n.replace(".cls", ".label")))
    )
    assert "label" in sample and "cls" not in sample


def test_rename_files_changes_the_sample_key(imagenet):
    """Renaming happens before grouping, so it can change a sample's key.

    Applying it after grouping would look right for extension rewrites and
    silently leave the key alone here.
    """
    sample = next(iter(wds.WebDataset(imagenet, shardshuffle=False, rename_files=lambda n: "x_" + n)))
    assert sample["__key__"] == "x_10"


def test_rename_files_regroups_and_reports_a_collision(imagenet):
    """Merging two keys puts two files under one extension, which must raise."""
    merge = lambda n: n.rpartition(".")[0][:-1] + "." + n.rpartition(".")[2]  # noqa: E731
    with pytest.raises(ValueError, match="duplicate"):
        list(wds.WebDataset(imagenet, shardshuffle=False, rename_files=merge))


def test_brace_expansion_names_every_shard():
    shards = wds.SimpleShardList("data-{000000..000009}.tar")
    assert len(shards) == 10
    assert next(iter(shards))["url"] == "data-000000.tar"


def test_separator_concatenates(imagenet):
    both = f"{imagenet}::{imagenet}"
    assert len(list(wds.WebDataset(both, shardshuffle=False))) == 94


# -- the explicit pipeline API --------------------------------------------


def test_explicit_pipeline(imagenet):
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(imagenet),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.to_tuple("png", "cls"),
        wds.batched(16),
    )
    batches = list(pipeline)
    assert len(batches) == 3
    assert isinstance(batches[0][1], np.ndarray)


def test_explicit_pipeline_with_splitters(imagenet):
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(imagenet),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.decode(),
    )
    assert len(list(pipeline)) == 47


def test_explicit_pipeline_with_a_python_stage(imagenet):
    pipeline = wds.DataPipeline(
        wds.SimpleShardList(imagenet),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(lambda s: {**s, "seen": True}),
    )
    samples = list(pipeline)
    assert len(samples) == 47 and all(s["seen"] for s in samples)


# -- lowering --------------------------------------------------------------


def test_a_whole_pipeline_lowers(imagenet):
    plan = wds.WebDataset(imagenet).shuffle(100).decode("rgb8").to_tuple("png", "cls").batched(8).explain()
    assert plan["python"] == [], "nothing should be left for Python"
    assert plan["native"] == ["read", "shuffle", "decode", "to_tuple", "batched"]


def test_a_python_stage_stops_the_lowering(imagenet):
    plan = wds.WebDataset(imagenet).decode().map(lambda s: s).to_tuple("png", "cls").explain()
    assert plan["native"] == ["read", "decode"]
    assert plan["python"] == ["map", "to_tuple"], "everything after the Python stage follows it"


def test_a_custom_collation_stays_in_python(imagenet):
    plan = wds.WebDataset(imagenet).decode().batched(8, collation_fn=lambda b: b).explain()
    assert plan["python"] == ["batched"]


# -- writing ---------------------------------------------------------------


def test_round_trip(written):
    samples = list(wds.WebDataset(written, shardshuffle=False).decode())
    assert len(samples) == 20
    assert samples[3]["cls"] == 3
    assert samples[3]["txt"] == "sample number 3"
    assert samples[3]["json"] == {"index": 3, "even": False}
    np.testing.assert_array_equal(samples[3]["npy"], np.arange(4, dtype=np.float32) * 3)


def test_shard_writer_rolls_over(tmp_path):
    pattern = str(tmp_path / "out-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=8) as writer:
        for i in range(20):
            writer.write({"__key__": f"k{i}", "txt": f"sample {i}"})
        assert writer.total == 20
    assert (tmp_path / "out-000000.tar").exists()
    assert (tmp_path / "out-000002.tar").exists()

    url = str(tmp_path / "out-{000000..000002}.tar")
    assert len(list(wds.WebDataset(url, shardshuffle=False))) == 20


def test_writer_refuses_a_sample_without_a_key(tmp_path):
    with pytest.raises(Exception), wds.TarWriter(str(tmp_path / "bad.tar")) as writer:
        writer.write({"txt": "no key here"})


def test_compressed_shards_round_trip(tmp_path):
    path = tmp_path / "out.tar.gz"
    with wds.TarWriter(str(path)) as writer:
        writer.write({"__key__": "k", "txt": "compressed"})
    assert path.read_bytes()[:2] == b"\x1f\x8b"
    assert next(iter(wds.WebDataset(str(path), shardshuffle=False).decode()))["txt"] == "compressed"


# -- error handling --------------------------------------------------------


def test_a_missing_shard_raises():
    with pytest.raises(Exception):
        list(wds.WebDataset("/no/such/shard.tar", shardshuffle=False))


def test_a_missing_shard_can_be_skipped(imagenet):
    url = f"/no/such/shard.tar::{imagenet}"
    samples = list(wds.WebDataset(url, shardshuffle=False, handler=wds.ignore_and_continue))
    assert len(samples) == 47


def test_to_tuple_reports_a_missing_field(imagenet):
    with pytest.raises(Exception):
        list(wds.WebDataset(imagenet, shardshuffle=False).to_tuple("nosuchfield"))


# -- the rest of the surface ----------------------------------------------


def test_tenbin_round_trip():
    arrays = [np.arange(6, dtype=np.float64).reshape(2, 3), np.ones((4,), dtype=np.float32)]
    encoded = wds.tenbin.encode_buffer(arrays)
    decoded = wds.tenbin.decode_buffer(encoded)
    assert len(decoded) == 2
    np.testing.assert_array_equal(decoded[0], arrays[0])
    np.testing.assert_array_equal(decoded[1], arrays[1])


def test_tenbin_shard_decodes():
    samples = list(wds.WebDataset(str(TESTDATA / "tendata.tar"), shardshuffle=False).decode())
    assert len(samples) == 100
    assert len(samples[0]["ten"]) == 2
    assert samples[0]["ten"][0].shape == (28, 28)


def test_gzipped_fields_decode():
    sample = next(iter(wds.WebDataset(str(TESTDATA / "compressed.tar"), shardshuffle=False).decode()))
    assert sample["txt.gz"] == "hello\n"


def test_base_plus_ext():
    assert wds.base_plus_ext("dir/a.seg.png") == ("dir/a", "seg.png")
    assert wds.base_plus_ext("noextension") == (None, None)


def test_handlers_are_the_documented_functions():
    assert wds.reraise_exception is not None
    assert wds.ignore_and_continue(ValueError("x")) is True
    assert wds.ignore_and_stop(ValueError("x")) is False


def test_worker_info_is_reported():
    rank, world, worker, workers = wds.utils.pytorch_worker_info()
    assert (rank, world, worker, workers) == (0, 1, 0, 1)


def test_the_documented_names_exist():
    missing = [name for name in wds.__all__ if not hasattr(wds, name)]
    assert missing == [], f"missing from the package: {missing}"
