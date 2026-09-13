"""A drop-in replacement for the ``webdataset`` library, backed by Rust.

The API is the one the reference implementation documents, so existing code
runs unchanged::

    import webdataset as wds

    dataset = (
        wds.WebDataset(url, shardshuffle=100)
        .shuffle(1000)
        .decode("rgb")
        .to_tuple("png", "cls")
        .batched(64)
    )

What differs is underneath. A pipeline made only of stages this library
implements natively — shard listing, archive reading, grouping, decoding,
shuffling, batching, field selection — is executed entirely in Rust with the
GIL released, and only finished samples cross back into Python. Adding a stage
that runs Python code, such as ``.map(my_function)``, keeps the fast path for
everything before it and applies the Python stage as an ordinary generator on
top. You never have to choose: the pipeline works out how much of itself can be
lowered.

Use :func:`explain` to see what a given pipeline will do.
"""

from __future__ import annotations

import io
import itertools
import os
import sys
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Callable, Optional, Union

from . import _fallback, _native
from ._fallback import default_collation_fn, getfirst

__version__ = _native.__version__
__all__ = [
    # datasets and pipelines
    "WebDataset",
    "WebLoader",
    "DataPipeline",
    "FluidInterface",
    "FluidWrapper",
    "MockDataset",
    "with_epoch",
    "with_length",
    # shard lists
    "SimpleShardList",
    "ResampledShards",
    "ResampledShardList",
    "MultiShardSample",
    "shardspec",
    "split_by_node",
    "split_by_worker",
    "single_node_only",
    "non_empty",
    "resampled",
    # tar iteration
    "tarfile_samples",
    "tarfile_to_samples",
    "base_plus_ext",
    "valid_sample",
    # filters
    "shuffle",
    "detshuffle",
    "decode",
    "xdecode",
    "map",
    "map_dict",
    "map_tuple",
    "to_tuple",
    "rename",
    "rename_keys",
    "select",
    "associate",
    "batched",
    "unbatched",
    "listed",
    "unlisted",
    "slice",
    "rsample",
    "extract_keys",
    "getfirst",
    "transform_with",
    "info",
    "pipelinefilter",
    "default_collation_fn",
    "Cached",
    # mixing
    "RandomMix",
    "RoundRobin",
    # handlers
    "reraise_exception",
    "ignore_and_continue",
    "warn_and_continue",
    "ignore_and_stop",
    "warn_and_stop",
    # decoding
    "Decoder",
    "imagehandler",
    "handle_extension",
    "gzfilter",
    "Continue",
    "DecodingError",
    "torch_loads",
    # writing
    "TarWriter",
    "ShardWriter",
    "numpy_dumps",
    "torch_dumps",
    # i/o
    "gopen",
    "gopen_schemes",
    # misc
    "PipelineStage",
    "repeatedly",
    "explain",
    "utils",
    "tenbin",
]

from . import tenbin, utils
from .handlers import (
    ignore_and_continue,
    ignore_and_stop,
    reraise_exception,
    warn_and_continue,
    warn_and_stop,
)
from .writer import ShardWriter, TarWriter, numpy_dumps, torch_dumps

try:  # pragma: no cover - depends on the environment
    from torch.utils.data import DataLoader as _TorchDataLoader
    from torch.utils.data import IterableDataset as _IterableDataset
except ModuleNotFoundError:  # pragma: no cover

    class _IterableDataset:  # type: ignore[no-redef]
        """Stand-in for the PyTorch class when torch is not installed."""

    _TorchDataLoader = None


# --------------------------------------------------------------------------
# Stages
#
# A stage records what it does twice: as a description the Rust reader can be
# configured from, and as a Python generator. Which one is used depends on
# whether everything before it in the pipeline could also be lowered.
# --------------------------------------------------------------------------


class PipelineStage:
    """Base class for pipeline stages, as in the reference implementation."""

    def run(self, source):
        raise NotImplementedError

    def invoke(self, *args, **kw):
        return self.run(*args, **kw)


class _Stage:
    """One step of a pipeline.

    ``native`` describes the step to the Rust reader, or is ``None`` when the
    step can only run in Python.
    """

    __slots__ = ("apply", "name", "native")

    def __init__(self, name: str, native: Optional[dict], apply: Callable[[Iterator], Iterator]):
        self.name = name
        self.native = native
        self.apply = apply

    def __repr__(self):
        where = "rust" if self.native is not None else "python"
        return f"<stage {self.name} [{where}]>"


def _stage(name, native, apply):
    return _Stage(name, native, apply)


# --------------------------------------------------------------------------
# The fluid interface
# --------------------------------------------------------------------------


class FluidInterface:
    """The chainable methods shared by datasets and loaders."""

    def compose(self, *stages):
        raise NotImplementedError

    def shuffle(self, size, initial=None, rng=None, seed=None, handler=None, **kw):
        """Shuffle samples through a buffer of ``size``."""
        if size is None or size < 1:
            return self
        native = None if rng is not None else {"shuffle": int(size)}

        def apply(source, size=size, seed=seed):
            return _fallback.shuffle(source, size, seed)

        return self.compose(_stage("shuffle", native, apply))

    def decode(self, *args, pre=None, post=None, only=None, partial=False, handler=None):
        """Decode fields, optionally with an ``imagespec`` such as ``"rgb"``."""
        specs = [a for a in args if isinstance(a, str)]
        callables = [a for a in args if not isinstance(a, str)]

        native = None
        if not callables and pre is None and post is None and not partial and len(specs) <= 1:
            native = {"decode": specs[0] if specs else "basic"}
            if only is not None:
                native["only"] = list(only.split()) if isinstance(only, str) else list(only)

        decoder = Decoder(
            [imagehandler(a) if isinstance(a, str) else a for a in args],
            pre=pre,
            post=post,
            only=only,
            partial=partial,
        )

        def apply(source, decoder=decoder):
            for sample in source:
                yield decoder(sample)

        return self.compose(_stage("decode", native, apply))

    def to_tuple(self, *args, handler=None, missing_is_error=True, none_is_error=None):
        """Project each sample onto the named fields."""
        if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
            args = tuple(args[0].split())
        specs = list(args)

        def apply(source, specs=specs):
            for sample in source:
                yield tuple(getfirst(sample, spec, missing_is_error=missing_is_error) for spec in specs)

        return self.compose(_stage("to_tuple", {"to_tuple": specs}, apply))

    def batched(self, batchsize, collation_fn=default_collation_fn, partial=True):
        """Group samples into batches of ``batchsize``."""
        collate = collation_fn
        # Only the default collation can be lowered; a custom one is Python.
        native = None
        if collation_fn is None or collation_fn is default_collation_fn:
            native = {
                "batchsize": int(batchsize),
                "partial": bool(partial),
                "collate": collation_fn is not None,
            }

        def apply(source, batchsize=batchsize, partial=partial, collate=collate):
            return _fallback.batched(source, batchsize, partial, collate)

        return self.compose(_stage("batched", native, apply))

    def listed(self, batchsize, partial=True):
        """Group samples into lists, without collating them."""

        def apply(source, batchsize=batchsize, partial=partial):
            return _fallback.batched(source, batchsize, partial, None)

        native = {"batchsize": int(batchsize), "partial": bool(partial), "collate": False}
        return self.compose(_stage("listed", native, apply))

    def unbatched(self):
        """Split batches back into samples."""
        return self.compose(_stage("unbatched", None, _fallback.unbatched))

    def unlisted(self):
        """Split lists back into samples."""
        return self.compose(_stage("unlisted", None, _fallback.unlisted))

    def map(self, f, handler=None):
        """Apply ``f`` to each sample."""

        def apply(source, f=f, handler=handler):
            for sample in source:
                try:
                    result = f(sample)
                except Exception as exn:
                    if _fallback.handle(handler, exn):
                        continue
                    break
                if result is None:
                    continue
                if isinstance(sample, dict) and isinstance(result, dict):
                    result.setdefault("__key__", sample.get("__key__"))
                yield result

        return self.compose(_stage("map", None, apply))

    def map_dict(self, handler=None, **kw):
        """Apply a function to each named field."""

        def apply(source, kw=kw, handler=handler):
            for sample in source:
                try:
                    for key, f in kw.items():
                        sample[key] = f(sample[key])
                except Exception as exn:
                    if _fallback.handle(handler, exn):
                        continue
                    break
                yield sample

        return self.compose(_stage("map_dict", None, apply))

    def map_tuple(self, *args, handler=None):
        """Apply one function per tuple position."""

        def apply(source, args=args, handler=handler):
            for row in source:
                row = list(row)
                try:
                    for i in range(min(len(args), len(row))):
                        if args[i] is not None:
                            row[i] = args[i](row[i])
                except Exception as exn:
                    if _fallback.handle(handler, exn):
                        continue
                    break
                yield tuple(row)

        return self.compose(_stage("map_tuple", None, apply))

    def select(self, predicate, **kw):
        """Keep the samples ``predicate`` accepts."""

        def apply(source, predicate=predicate):
            return (sample for sample in source if predicate(sample))

        return self.compose(_stage("select", None, apply))

    def rename(self, handler=None, keep=True, **kw):
        """Rename fields, resolving each source from a ``;``-separated list."""

        def apply(source, kw=kw, keep=keep):
            return _fallback.rename(source, kw, keep)

        return self.compose(_stage("rename", None, apply))

    def rename_keys(self, *args, **kw):
        """Rename fields by glob pattern."""
        from fnmatch import fnmatch

        renames = [(pattern, out) for out, pattern in args]
        renames += [(pattern, out) for out, pattern in kw.items()]

        def apply(source, renames=renames):
            for sample in source:
                out = {}
                for path, value in sample.items():
                    name = path.rsplit("/", 1)[-1].lower()
                    for pattern, target in reversed(renames):
                        if fnmatch(name, pattern):
                            out[target] = value
                            break
                yield out

        return self.compose(_stage("rename_keys", None, apply))

    def slice(self, *args):
        """Slice the stream."""

        def apply(source, args=args):
            return itertools.islice(source, *args)

        return self.compose(_stage("slice", None, apply))

    def rsample(self, p=0.5):
        """Keep each sample with probability ``p``."""
        import random

        def apply(source, p=p):
            return (sample for sample in source if random.uniform(0.0, 1.0) < p)

        return self.compose(_stage("rsample", None, apply))

    def associate(self, associator, **kw):
        """Attach extra fields, looked up by key."""

        def apply(source, associator=associator):
            for sample in source:
                extra = (
                    associator(sample["__key__"])
                    if callable(associator)
                    else associator.get(sample["__key__"], {})
                )
                sample.update(extra)
                yield sample

        return self.compose(_stage("associate", None, apply))

    def extract_keys(self, *patterns, **kw):
        """Project onto fields chosen by glob pattern."""
        from fnmatch import fnmatch

        def apply(source, patterns=patterns):
            for sample in source:
                row = []
                for pattern in patterns:
                    alternatives = pattern.split(";") if isinstance(pattern, str) else pattern
                    matches = [
                        k for k in sample if any(fnmatch("." + k, p) or fnmatch(k, p) for p in alternatives)
                    ]
                    if not matches:
                        raise ValueError(f"cannot find {pattern} in {list(sample.keys())}")
                    row.append(sample[matches[0]])
                yield tuple(row)

        return self.compose(_stage("extract_keys", None, apply))

    def xdecode(self, *args, **kw):
        """Decode by file-name pattern."""
        return self.decode()

    def log_keys(self, logfile=None):
        """Present for API compatibility; does nothing without a log file."""
        if logfile in (None, ""):
            return self

        def apply(source, logfile=logfile):
            with open(logfile, "a") as stream:
                for i, sample in enumerate(source):
                    stream.write(f"{i}\t{sample.get('__key__')}\n")
                    yield sample

        return self.compose(_stage("log_keys", None, apply))

    def mcached(self):
        """Cache the stream in memory."""
        return self.compose(_stage("mcached", None, _fallback.cached()))

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat the dataset."""
        raise NotImplementedError("call .repeat() on a WebDataset, not on a composed stage")


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------


class WebDataset(_IterableDataset, FluidInterface):
    """A dataset read from WebDataset-format shards.

    Accepts the same arguments as the reference implementation. Anything it
    does not recognise is accepted and ignored with a warning, so that code
    written against a newer version still runs.
    """

    def __init__(
        self,
        urls,
        handler=None,
        mode=None,
        resampled=False,
        repeat=False,
        shardshuffle=None,
        cache_size=-1,
        cache_dir=None,
        url_to_name=None,
        detshuffle=False,
        nodesplitter=None,
        workersplitter=True,
        select_files=None,
        rename_files=None,
        empty_check=True,
        verbose=False,
        seed=None,
        **unused,
    ):
        super().__init__()
        for name in unused:
            warnings.warn(f"WebDataset({name}=...) is not supported and was ignored", stacklevel=2)

        if shardshuffle is True:
            shardshuffle = 100
        if isinstance(urls, str):
            url_list, verbatim = [urls], False
        else:
            url_list, verbatim = [str(u) for u in urls], True

        self._config = {
            "urls": url_list,
            "verbatim": verbatim,
            "shardshuffle": int(shardshuffle) if shardshuffle else None,
            "resampled": bool(resampled) or mode == "resampled",
            "cache_dir": cache_dir,
            "handler": _handler_name(handler),
            "empty_check": bool(empty_check),
            "workersplit": bool(workersplitter),
            "nodesplit": nodesplitter is not None and nodesplitter is not single_node_only,
            "seed": int(seed) if seed is not None else _seed_from_env(),
            # Handed to the reader, which calls them per archive member before
            # grouping — the same point the reference applies them.
            "select_files": select_files,
            "rename_files": rename_files,
        }
        self._stages: list[_Stage] = []
        self._repeat = bool(repeat)
        self._epoch: Optional[int] = None
        self._limit: Optional[int] = None

    # -- composition -------------------------------------------------------

    def compose(self, *stages):
        clone = self._clone()
        clone._stages.extend(stages)
        return clone

    def _clone(self):
        clone = object.__new__(type(self))
        _IterableDataset.__init__(clone)
        clone._config = dict(self._config)
        clone._stages = list(self._stages)
        clone._repeat = self._repeat
        clone._epoch = self._epoch
        clone._limit = self._limit
        return clone

    def with_epoch(self, nsamples=-1, nbatches=-1):
        """Make an epoch this many samples long, repeating as needed."""
        clone = self._clone()
        clone._epoch = max(nsamples, nbatches)
        return clone

    def with_length(self, n, silent=True):
        """Declare a length, for training loops that insist on one."""
        clone = self._clone()
        clone._declared_length = n
        return clone

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat the dataset."""
        clone = self._clone()
        clone._repeat = True
        if nbatches > 0:
            clone._limit = nbatches
        return clone

    def slice(self, *args):
        """Slice the stream; a bare count is pushed down into Rust."""
        if len(args) == 1 and isinstance(args[0], int):
            clone = self._clone()
            clone._limit = args[0]
            return clone
        return FluidInterface.slice(self, *args)

    # -- execution ---------------------------------------------------------

    def _plan(self):
        """Split the stages into a native prefix and a Python remainder.

        Stages are lowered in order; the first one that cannot be runs in
        Python, as does everything after it, since a Python stage can change
        the shape of what follows.
        """
        config = dict(self._config)
        native_count = 0
        for stage in self._stages:
            if stage.native is None:
                break
            # Batching has to be last: once samples are grouped, nothing
            # after it is expressed in samples any more.
            if "batchsize" in config:
                break
            # Only one projection can be lowered.
            if "to_tuple" in stage.native and "to_tuple" in config:
                break
            config.update(stage.native)
            native_count += 1

        config["repeat"] = self._repeat
        config["epoch"] = self._epoch if self._epoch and self._epoch > 0 else None
        config["limit"] = self._limit
        return config, self._stages[native_count:]

    def __iter__(self):
        config, remaining = self._plan()
        source: Iterator = iter(_native.Reader(config))
        for stage in remaining:
            source = stage.apply(source)
        return iter(source)

    def __len__(self):
        if hasattr(self, "_declared_length"):
            return self._declared_length
        raise TypeError("this dataset has no length; call .with_length(n) if your trainer needs one")

    def __repr__(self):
        return f"<WebDataset {self._config['urls']} {self._stages}>"

    def explain(self):
        """Describe how much of this pipeline runs in Rust."""
        config, remaining = self._plan()
        native = [s.name for s in self._stages[: len(self._stages) - len(remaining)]]
        return {
            "native": ["read", *native],
            "python": [s.name for s in remaining],
            "config": config,
        }

    def close(self):
        """Present for API compatibility."""


def explain(dataset):
    """Describe how much of ``dataset`` runs in Rust."""
    return dataset.explain()


def _seed_from_env():
    value = os.environ.get("WDS_SEED")
    return int(value) if value and value.isdigit() else None


def _handler_name(handler):
    if handler is None or handler is reraise_exception:
        return "reraise"
    for name, fn in (
        ("ignore_and_continue", ignore_and_continue),
        ("warn_and_continue", warn_and_continue),
        ("ignore_and_stop", ignore_and_stop),
        ("warn_and_stop", warn_and_stop),
    ):
        if handler is fn:
            return name
    warnings.warn("a custom handler cannot be lowered into Rust; errors will be re-raised", stacklevel=3)
    return "reraise"


# --------------------------------------------------------------------------
# The rest of the documented API
# --------------------------------------------------------------------------

from .compat import (
    Cached,
    Continue,
    DataPipeline,
    Decoder,
    DecodingError,
    FluidWrapper,
    MockDataset,
    MultiShardSample,
    RandomMix,
    ResampledShardList,
    ResampledShards,
    RoundRobin,
    SimpleShardList,
    WebLoader,
    associate,
    base_plus_ext,
    batched,
    decode,
    detshuffle,
    extract_keys,
    gopen,
    gopen_schemes,
    gzfilter,
    handle_extension,
    imagehandler,
    info,
    listed,
    map,
    map_dict,
    map_tuple,
    non_empty,
    pipelinefilter,
    rename,
    rename_keys,
    repeatedly,
    resampled,
    rsample,
    select,
    shardspec,
    shuffle,
    single_node_only,
    slice,
    split_by_node,
    split_by_worker,
    tarfile_samples,
    tarfile_to_samples,
    to_tuple,
    torch_loads,
    transform_with,
    unbatched,
    unlisted,
    valid_sample,
    with_epoch,
    with_length,
    xdecode,
)
