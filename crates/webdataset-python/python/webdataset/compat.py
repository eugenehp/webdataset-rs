"""The explicit pipeline API, and the pieces the fluid one is built from.

The reference implementation offers two interfaces: the fluid one on
:class:`~webdataset.WebDataset`, and the explicit :class:`DataPipeline`, whose
stages are written out one per line::

    dataset = wds.DataPipeline(
        wds.SimpleShardList(url),
        wds.shuffle(100),
        wds.split_by_worker,
        wds.tarfile_to_samples(),
        wds.shuffle(1000),
        wds.decode("rgb"),
        wds.to_tuple("png", "cls"),
        wds.batched(16),
    )

Both are supported, and both are lowered into Rust as far as they can be. A
pipeline that starts with a shard list followed by ``tarfile_to_samples`` is
recognised and handed to the native reader wholesale; the stages after it are
lowered one at a time until one is reached that runs Python code.
"""

from __future__ import annotations

import itertools
import random
import sys
from collections.abc import Iterator

from . import _fallback, _native
from ._fallback import default_collation_fn, getfirst
from .handlers import reraise_exception
from .utils import PipelineStage, base_plus_ext, pytorch_worker_info, repeatedly

__all__ = [
    "DataPipeline",
    "FluidWrapper",
    "WebLoader",
    "MockDataset",
    "SimpleShardList",
    "ResampledShards",
    "ResampledShardList",
    "MultiShardSample",
    "RandomMix",
    "RoundRobin",
    "Cached",
    "Decoder",
    "DecodingError",
    "Continue",
    "imagehandler",
    "handle_extension",
    "gzfilter",
    "torch_loads",
    "shardspec",
    "split_by_node",
    "split_by_worker",
    "single_node_only",
    "non_empty",
    "resampled",
    "tarfile_samples",
    "tarfile_to_samples",
    "valid_sample",
    "getfirst",
    "transform_with",
    "default_collation_fn",
    "pipelinefilter",
    "gopen",
    "gopen_schemes",
    "with_epoch",
    "with_length",
    "repeatedly",
    "base_plus_ext",
]


# --------------------------------------------------------------------------
# Descriptions of stages
#
# Every filter below returns a `Filter`, which carries both a description the
# native reader can be configured from and a Python generator that does the
# same thing. Which one runs is decided when the pipeline is iterated.
# --------------------------------------------------------------------------


class Filter:
    """One stage of a pipeline."""

    def __init__(self, name, native, apply):
        self.name = name
        self.native = native
        self.apply = apply

    def __call__(self, source):
        return self.apply(source)

    def __repr__(self):
        return f"<{self.name} [{'rust' if self.native is not None else 'python'}]>"


def pipelinefilter(f):
    """Curry all but the first argument, as the reference implementation does."""

    def curried(*args, **kw):
        def apply(source):
            return f(source, *args, **kw)

        return Filter(getattr(f, "__name__", "filter"), None, apply)

    curried.__name__ = getattr(f, "__name__", "filter")
    return curried


# --------------------------------------------------------------------------
# Shard lists
# --------------------------------------------------------------------------


class ShardList:
    """A description of where shards come from, recognised by the pipeline."""

    def __init__(self, urls, *, resampled=False, seed=None):
        if isinstance(urls, str):
            self.urls, self.verbatim = [urls], False
        else:
            self.urls, self.verbatim = [str(u) for u in urls], True
        self.resampled = resampled
        self.seed = seed

    def __iter__(self):
        """Yield ``dict(url=...)``, as the reference implementation does."""
        expanded = []
        for pattern in self.urls:
            expanded.extend([pattern] if self.verbatim else _native.expand_urls(pattern))
        if self.resampled:
            rng = random.Random(self.seed)
            while True:
                yield dict(url=rng.choice(expanded))
        for url in expanded:
            yield dict(url=url)

    def __len__(self):
        if self.verbatim:
            return len(self.urls)
        return sum(len(_native.expand_urls(p)) for p in self.urls)


class SimpleShardList(ShardList):
    """A fixed list of shard URLs."""

    def __init__(self, urls, seed=None):
        super().__init__(urls, resampled=False, seed=seed)


class ResampledShards(ShardList):
    """Shards drawn with replacement, forever."""

    def __init__(self, urls, nshards=sys.maxsize, seed=0, deterministic=False, **kw):
        super().__init__(urls, resampled=True, seed=seed)
        self.nshards = nshards


ResampledShardList = ResampledShards


def shardspec(spec):
    """Build a shard list from a specification string."""
    if spec.endswith(".yaml"):
        return MultiShardSample(spec)
    return SimpleShardList(spec)


class MultiShardSample:
    """Several shard sources mixed by a YAML specification."""

    def __init__(self, fname):
        import yaml

        if isinstance(fname, dict):
            spec = fname
        else:
            with open(fname) as stream:
                spec = yaml.safe_load(stream)
        prefix = spec.get("prefix", "")
        buckets = spec.get("buckets", "")
        self.urls = []
        for dataset in spec["datasets"]:
            bucket = dataset.get("buckets", buckets)
            bucket = bucket[0] if isinstance(bucket, list) and bucket else bucket
            shards = dataset["shards"]
            shards = [shards] if isinstance(shards, str) else shards
            for pattern in shards:
                for url in _native.braceexpand(pattern):
                    joined = f"{bucket}/{url}" if bucket else url
                    self.urls.append(f"{prefix}{joined}")

    def __iter__(self):
        for url in self.urls:
            yield dict(url=url)


def split_by_node(src, group=None):
    """Keep only this distributed rank's shards."""
    rank, world_size, _, _ = pytorch_worker_info(group=group)
    if world_size > 1:
        yield from itertools.islice(src, rank, None, world_size)
    else:
        yield from src


def split_by_worker(src):
    """Keep only this loader worker's shards."""
    _, _, worker, num_workers = pytorch_worker_info()
    if num_workers > 1:
        yield from itertools.islice(src, worker, None, num_workers)
    else:
        yield from src


def single_node_only(src, group=None):
    """Refuse to run distributed without an explicit splitter."""
    _, world_size, _, _ = pytorch_worker_info(group=group)
    if world_size > 1:
        raise ValueError("add an explicit nodesplitter for multi-node training")
    yield from src


def non_empty(src):
    """Fail if the stream turns out to be empty."""
    count = 0
    for sample in src:
        yield sample
        count += 1
    if count == 0:
        raise ValueError("pipeline stage received no data at all and this was declared as an error")


def resampled_(src, n=sys.maxsize):
    """Draw from ``src`` with replacement."""
    items = list(src)
    rng = random.Random()
    for _ in range(n):
        yield rng.choice(items)


resampled = pipelinefilter(resampled_)


# --------------------------------------------------------------------------
# Reading archives
# --------------------------------------------------------------------------


def valid_sample(sample):
    """Whether a sample should be passed downstream."""
    return (
        sample is not None
        and isinstance(sample, dict)
        and len(sample) > 0
        and not sample.get("__bad__", False)
    )


def tarfile_samples(src, handler=reraise_exception, select_files=None, rename_files=None):
    """Turn a stream of ``dict(url=...)`` into a stream of samples."""
    for source in src:
        url = source["url"] if isinstance(source, dict) else source
        config = {"urls": [url], "verbatim": True, "empty_check": False, "workersplit": False}
        yield from _native.Reader(config)


def tarfile_to_samples(handler=reraise_exception, select_files=None, rename_files=None):
    """The curried form of :func:`tarfile_samples`."""

    def apply(source):
        return tarfile_samples(source, handler=handler, select_files=select_files, rename_files=rename_files)

    # `native` is the empty dict because reading is what the native reader does
    # by default; the stage exists only to mark where it happens.
    return Filter("tarfile_to_samples", {}, apply)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


class Continue:
    """Ask the decoder to carry on with rewritten contents."""

    def __init__(self, key, data):
        self.key, self.data = key, data


class DecodingError(Exception):
    """Raised when a field cannot be decoded."""

    def __init__(self, url=None, key=None, k=None, sample=None):
        super().__init__(f"cannot decode {k} of {key} from {url}")
        self.url, self.key, self.k, self.sample = url, key, k, sample


class ImageHandler:
    """Decode images according to an ``imagespec``."""

    def __init__(self, imagespec, extensions=None):
        self.imagespec = imagespec.lower()

    def __call__(self, key, data):
        raise NotImplementedError("image handlers are applied by the decoder, not called directly")


def imagehandler(imagespec, extensions=None):
    """Build an image handler for an ``imagespec`` such as ``"rgb"``."""
    return ImageHandler(imagespec, extensions)


def handle_extension(extensions, f):
    """Build a handler that fires for the given extensions."""

    def handler(key, data):
        extension = key.lower().split(".")
        for target in extensions.lower().split():
            parts = target.split(".")
            if len(parts) <= len(extension) and extension[-len(parts) :] == parts:
                return f(data)
        return None

    return handler


def gzfilter(key, data):
    """Decompress ``.gz`` fields; applied natively by the decoder."""
    import gzip

    if not key.endswith(".gz"):
        return None
    return Continue(key[:-3], gzip.decompress(data))


def torch_loads(data):
    """Load a ``torch.save`` payload."""
    import io

    import torch

    return torch.load(io.BytesIO(data), map_location="cpu")


class Decoder:
    """Decode samples, with the same defaults as the reference implementation.

    Decoding happens in Rust. Handlers that are plain Python callables are
    applied afterwards, over whatever Rust left as ``bytes``.
    """

    def __init__(self, handlers=None, pre=None, post=None, only=None, partial=False):
        handlers = list(handlers or [])
        specs = [h.imagespec for h in handlers if isinstance(h, ImageHandler)]
        self.imagespec = specs[0] if specs else "basic"
        self.callables = [h for h in handlers if not isinstance(h, ImageHandler)]
        self.pre = list(pre or [])
        self.post = list(post or [])
        self.only = only.split() if isinstance(only, str) else (list(only) if only else None)
        self.partial = partial

    def decode(self, sample):
        """Decode a whole sample."""
        return self(sample)

    def __call__(self, sample):
        raw = {k: v for k, v in sample.items() if isinstance(v, (bytes, bytearray))}
        rest = {k: v for k, v in sample.items() if k not in raw}
        decoded = _native.decode_sample(raw, self.imagespec, self.only)
        decoded.update(rest)

        extra = self.pre + self.callables + self.post
        if not extra:
            return decoded
        for key, value in list(decoded.items()):
            if key.startswith("__") or not isinstance(value, (bytes, bytearray)):
                continue
            for handler in extra:
                result = handler("." + key, value)
                if isinstance(result, Continue):
                    key, value = result.key, result.data
                    continue
                if result is not None:
                    decoded[key] = result
                    break
        return decoded


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def transform_with(sample, transformers):
    """Apply one function per position."""
    if not transformers:
        return sample
    result = list(sample)
    for i, f in enumerate(transformers):
        if f is not None:
            result[i] = f(sample[i])
    return result


def _filter(name, native, make):
    """Build a curried filter that records both forms."""

    def curried(*args, **kw):
        return Filter(name, native(*args, **kw) if native else None, make(*args, **kw))

    curried.__name__ = name
    return curried


def _shuffle_apply(size, initial=None, rng=None, seed=None, handler=None, **kw):
    def apply(source):
        return _fallback.shuffle(source, size, seed)

    return apply


shuffle = _filter(
    "shuffle", lambda size, **kw: {"shuffle": int(size)} if size and size >= 1 else None, _shuffle_apply
)


def detshuffle(bufsize=1000, initial=100, seed=0, epoch=-1):
    """Deterministic shuffling."""
    return Filter(
        "detshuffle", {"shuffle": int(bufsize), "seed": int(seed)}, _shuffle_apply(bufsize, seed=seed)
    )


def _decode_native(*args, **kw):
    specs = [a for a in args if isinstance(a, str)]
    if any(not isinstance(a, str) for a in args) or kw.get("pre") or kw.get("post") or kw.get("partial"):
        return None
    native = {"decode": specs[0] if specs else "basic"}
    only = kw.get("only")
    if only is not None:
        native["only"] = only.split() if isinstance(only, str) else list(only)
    return native


def _decode_apply(*args, **kw):
    decoder = Decoder(
        [imagehandler(a) if isinstance(a, str) else a for a in args],
        pre=kw.get("pre"),
        post=kw.get("post"),
        only=kw.get("only"),
        partial=kw.get("partial", False),
    )

    def apply(source):
        for sample in source:
            yield decoder(sample)

    return apply


decode = _filter("decode", _decode_native, _decode_apply)


def _to_tuple_native(*args, **kw):
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = tuple(args[0].split())
    return {"to_tuple": list(args)}


def _to_tuple_apply(*args, **kw):
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = tuple(args[0].split())
    missing_is_error = kw.get("missing_is_error", True)

    def apply(source):
        for sample in source:
            yield tuple(getfirst(sample, spec, missing_is_error=missing_is_error) for spec in args)

    return apply


to_tuple = _filter("to_tuple", _to_tuple_native, _to_tuple_apply)


def _batched_native(batchsize, collation_fn=default_collation_fn, partial=True):
    if collation_fn is not default_collation_fn and collation_fn is not None:
        return None
    return {"batchsize": int(batchsize), "partial": bool(partial), "collate": collation_fn is not None}


def _batched_apply(batchsize, collation_fn=default_collation_fn, partial=True):
    def apply(source):
        return _fallback.batched(source, batchsize, partial, collation_fn)

    return apply


batched = _filter("batched", _batched_native, _batched_apply)


def _listed_apply(batchsize, partial=True):
    def apply(source):
        return _fallback.batched(source, batchsize, partial, None)

    return apply


listed = _filter(
    "listed",
    lambda batchsize, partial=True: {"batchsize": int(batchsize), "partial": bool(partial), "collate": False},
    _listed_apply,
)


def _simple(name):
    """Build a Python-only filter from a generator function."""

    def decorator(f):
        def curried(*args, **kw):
            def apply(source):
                return f(source, *args, **kw)

            return Filter(name, None, apply)

        curried.__name__ = name
        return curried

    return decorator


@_simple("map")
def map(source, f, handler=reraise_exception):
    """Apply ``f`` to each sample."""
    for sample in source:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            break
        if result is None:
            continue
        if isinstance(sample, dict) and isinstance(result, dict):
            result.setdefault("__key__", sample.get("__key__"))
        yield result


@_simple("map_dict")
def map_dict(source, handler=reraise_exception, **kw):
    """Apply a function to each named field."""
    for sample in source:
        for key, f in kw.items():
            sample[key] = f(sample[key])
        yield sample


@_simple("map_tuple")
def map_tuple(source, *args, handler=reraise_exception):
    """Apply one function per tuple position."""
    for row in source:
        row = list(row)
        for i in range(min(len(args), len(row))):
            if args[i] is not None:
                row[i] = args[i](row[i])
        yield tuple(row)


@_simple("select")
def select(source, predicate, **kw):
    """Keep the samples ``predicate`` accepts."""
    for sample in source:
        if predicate(sample):
            yield sample


@_simple("rename")
def rename(source, handler=reraise_exception, keep=True, **kw):
    """Rename fields."""
    yield from _fallback.rename(source, kw, keep)


@_simple("rename_keys")
def rename_keys(source, *args, keep_unselected=False, must_match=True, duplicate_is_error=True, **kw):
    """Rename fields by glob pattern."""
    from fnmatch import fnmatch

    renames = [(pattern, out) for out, pattern in args] + [(pattern, out) for out, pattern in kw.items()]
    for sample in source:
        out = {}
        for path, value in sample.items():
            name = path.rsplit("/", 1)[-1].lower()
            for pattern, target in reversed(renames):
                if fnmatch(name, pattern):
                    out[target] = value
                    break
            else:
                if keep_unselected:
                    out[path] = value
        yield out


@_simple("associate")
def associate(source, associator, **kw):
    """Attach extra fields, looked up by key."""
    for sample in source:
        extra = (
            associator(sample["__key__"]) if callable(associator) else associator.get(sample["__key__"], {})
        )
        sample.update(extra)
        yield sample


@_simple("unbatched")
def unbatched(source):
    """Split batches back into samples."""
    yield from _fallback.unbatched(source)


@_simple("unlisted")
def unlisted(source):
    """Split lists back into samples."""
    for group in source:
        yield from group


@_simple("rsample")
def rsample(source, p=0.5):
    """Keep each sample with probability ``p``."""
    for sample in source:
        if random.uniform(0.0, 1.0) < p:
            yield sample


@_simple("slice")
def slice(source, *args):
    """Slice the stream."""
    yield from itertools.islice(source, *args)


@_simple("extract_keys")
def extract_keys(source, *patterns, duplicate_is_error=True, ignore_missing=False):
    """Project onto fields chosen by glob pattern."""
    from fnmatch import fnmatch

    for sample in source:
        row = []
        for pattern in patterns:
            alternatives = pattern.split(";") if isinstance(pattern, str) else pattern
            matches = [k for k in sample if any(fnmatch("." + k, p) or fnmatch(k, p) for p in alternatives)]
            if not matches:
                if ignore_missing:
                    continue
                raise ValueError(f"cannot find {pattern} in {list(sample.keys())}")
            row.append(sample[matches[0]])
        yield tuple(row)


@_simple("xdecode")
def xdecode(source, *args, **kw):
    """Decode by file-name pattern."""
    decoder = Decoder([])
    for sample in source:
        yield decoder(sample)


@_simple("info")
def info(source, fmt=None, n=3, every=-1, width=50, stream=sys.stderr, name=""):
    """Print the first few samples as they pass."""
    for i, sample in enumerate(source):
        if i < n or (every > 0 and (i + 1) % every == 0):
            print("---", name, file=stream)
            for k, v in sample.items():
                print(k, repr(v)[:width], file=stream)
        yield sample


class Cached(PipelineStage):
    """Cache the stream in memory."""

    def __init__(self):
        self.cached = None

    def run(self, source):
        if self.cached is not None:
            yield from self.cached
            return
        collected = []
        for sample in source:
            collected.append(sample)
            yield sample
        self.cached = collected


# --------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------


class RoundRobin:
    """Take from each dataset in turn."""

    def __init__(self, datasets, longest=False):
        self.datasets, self.longest = datasets, longest

    def __iter__(self):
        sources = [iter(d) for d in self.datasets]
        i = 0
        while sources:
            i %= len(sources)
            try:
                yield next(sources[i])
                i += 1
            except StopIteration:
                if not self.longest:
                    return
                del sources[i]


class RandomMix:
    """Draw from several datasets at given probabilities."""

    def __init__(self, datasets, probs=None, longest=False):
        self.datasets, self.probs, self.longest = datasets, probs, longest

    def __iter__(self):
        import numpy as np

        sources = [iter(d) for d in self.datasets]
        probs = list(self.probs) if self.probs else [1.0] * len(sources)
        while sources:
            cum = (np.array(probs) / np.sum(probs)).cumsum()
            i = int(np.searchsorted(cum, random.random()))
            try:
                yield next(sources[i])
            except StopIteration:
                if not self.longest:
                    return
                del sources[i]
                del probs[i]


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------


class DataPipeline:
    """A source followed by a list of stages.

    A pipeline that begins with a shard list and ``tarfile_to_samples`` is
    recognised and executed natively, with the following stages lowered until
    one is reached that runs Python code.
    """

    def __init__(self, *stages, **kw):
        self.pipeline = []
        for stage in stages:
            if stage is None:
                continue
            if isinstance(stage, list):
                self.pipeline.extend(stage)
            else:
                self.pipeline.append(stage)
        self.repetitions = 1
        self.nsamples = -1

    def append(self, stage):
        """Append a stage in place."""
        self.pipeline.append(stage)

    def compose(self, *stages):
        """Append stages to a copy."""
        clone = DataPipeline(*self.pipeline)
        clone.repetitions, clone.nsamples = self.repetitions, self.nsamples
        clone.pipeline.extend(stages)
        return clone

    def stage(self, i):
        """Return stage ``i``."""
        return self.pipeline[i]

    def with_epoch(self, nsamples=-1, nbatches=-1):
        """Make an epoch this many samples long."""
        self.repetitions = sys.maxsize
        self.nsamples = max(nsamples, nbatches)
        return self

    def with_length(self, n, silent=True):
        """Declare a length."""
        self._declared_length = n
        return self

    def repeat(self, nepochs=-1, nbatches=-1):
        """Repeat the pipeline."""
        self.repetitions = nepochs if nepochs > 0 else sys.maxsize
        self.nsamples = nbatches
        return self

    def __len__(self):
        if hasattr(self, "_declared_length"):
            return self._declared_length
        raise TypeError("this pipeline has no length; call .with_length(n) if your trainer needs one")

    def _plan(self):
        """Work out how much of the pipeline can run natively."""
        stages = list(self.pipeline)
        if not stages or not isinstance(stages[0], ShardList):
            return None, stages

        source = stages[0]
        rest = stages[1:]
        # The reader does its own splitting, so a splitter here is redundant.
        while rest and rest[0] in (split_by_worker, split_by_node, single_node_only):
            rest = rest[1:]

        config = {
            "urls": source.urls,
            "verbatim": source.verbatim,
            "resampled": source.resampled,
            "empty_check": False,
            "workersplit": False,
            "nodesplit": False,
        }
        if source.seed is not None:
            config["seed"] = int(source.seed)

        # A shard shuffle placed before the reader becomes `shardshuffle`.
        while rest and isinstance(rest[0], Filter) and rest[0].name in ("shuffle", "detshuffle"):
            if rest[0].native is None:
                break
            config["shardshuffle"] = rest[0].native["shuffle"]
            rest = rest[1:]
            while rest and rest[0] in (split_by_worker, split_by_node, single_node_only):
                rest = rest[1:]

        if not rest or not (isinstance(rest[0], Filter) and rest[0].name == "tarfile_to_samples"):
            return None, stages
        rest = rest[1:]

        lowered = 0
        for stage in rest:
            if not isinstance(stage, Filter) or stage.native is None:
                break
            if "batchsize" in config:
                break
            if "to_tuple" in stage.native and "to_tuple" in config:
                break
            config.update(stage.native)
            lowered += 1
        return config, rest[lowered:]

    def _iterator(self):
        config, remaining = self._plan()
        if config is not None:
            source: Iterator = iter(_native.Reader(config))
        else:
            source = self._invoke(self.pipeline[0])
            remaining = self.pipeline[1:]
        for stage in remaining:
            source = self._apply(stage, source)
        return iter(source)

    @staticmethod
    def _invoke(stage):
        if callable(stage) and not isinstance(stage, Filter):
            return iter(stage())
        return iter(stage)

    @staticmethod
    def _apply(stage, source):
        if isinstance(stage, Filter):
            return stage(source)
        if isinstance(stage, PipelineStage):
            return stage.run(source)
        if callable(stage):
            return stage(source)
        raise ValueError(f"{stage}: not a valid pipeline stage")

    def __iter__(self):
        if self.repetitions == 1:
            return self._iterator()

        def repeated():
            for _ in range(self.repetitions):
                count = 0
                for sample in self._iterator():
                    yield sample
                    count += 1
                if count == 0:
                    break

        stream = repeated()
        return itertools.islice(stream, self.nsamples) if self.nsamples > 0 else stream

    def explain(self):
        """Describe how much of this pipeline runs in Rust."""
        config, remaining = self._plan()
        return {
            "native": ["read"]
            + [
                k
                for k in ("shardshuffle", "shuffle", "decode", "to_tuple", "batchsize")
                if config and k in config
            ],
            "python": [getattr(s, "name", repr(s)) for s in remaining],
            "config": config,
        }

    def close(self):
        """Present for API compatibility."""


class FluidWrapper(DataPipeline):
    """A pipeline with the fluid interface bolted on."""

    def __init__(self, initial):
        super().__init__(initial)


class WebLoader(DataPipeline):
    """A thin wrapper over ``torch.utils.data.DataLoader``."""

    def __init__(self, *args, **kw):
        try:
            from torch.utils.data import DataLoader
        except ModuleNotFoundError as exn:  # pragma: no cover
            raise ModuleNotFoundError("WebLoader needs PyTorch installed") from exn
        super().__init__(DataLoader(*args, **kw))


class MockDataset:
    """Yield the same sample over and over, for benchmarking."""

    def __init__(self, sample, length):
        self.sample, self.length = sample, length

    def __iter__(self):
        for _ in range(self.length):
            yield self.sample


class with_epoch:
    """Impose an epoch boundary on an iterable."""

    def __init__(self, dataset, length):
        self.dataset, self.length = dataset, length
        self.source = None

    def __iter__(self):
        if self.source is None:
            self.source = iter(self.dataset)
        for _ in range(self.length):
            try:
                yield next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                try:
                    yield next(self.source)
                except StopIteration:
                    return


class with_length:
    """Declare a length for an iterable."""

    def __init__(self, dataset, length):
        self.dataset, self.length = dataset, length

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return self.length


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

#: The URL schemes the reader understands; present for API compatibility.
gopen_schemes = {
    "__default__": None,
    "pipe": None,
    "http": None,
    "https": None,
    "gs": None,
    "ais": None,
    "hf": None,
    "sftp": None,
    "ftps": None,
    "scp": None,
}


def gopen(url, mode="rb", bufsize=8192, **kw):
    """Open a URL.

    Local paths and ``pipe:`` URLs are handled here; anything else is read by
    the native reader when a shard is opened.
    """
    import subprocess

    if url == "-":
        return sys.stdin.buffer if mode == "rb" else sys.stdout.buffer
    if url.startswith("pipe:"):
        command = url[5:]
        if mode[0] == "r":
            return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout
        return subprocess.Popen(command, shell=True, stdin=subprocess.PIPE).stdin
    if "://" in url and not url.startswith("file://"):
        command = ["curl", "-s", "-L", url]
        if mode[0] == "r":
            return subprocess.Popen(command, stdout=subprocess.PIPE).stdout
        raise ValueError(f"{url}: cannot write")
    path = url[len("file://") :] if url.startswith("file://") else url
    return open(path, mode)
