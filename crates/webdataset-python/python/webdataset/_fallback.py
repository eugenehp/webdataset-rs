"""The Python implementations of stages that could not be lowered into Rust.

Every stage exists twice: once as a description the native reader is configured
from, and once as a generator here. A pipeline runs as much of itself natively
as it can and falls back to this module for the rest, so adding one Python
``.map()`` costs you that stage rather than the whole pipeline.

Nothing here imports from the package root, which is what keeps the import
graph acyclic.
"""

from __future__ import annotations

import random

__all__ = [
    "getfirst",
    "default_collation_fn",
    "handle",
    "shuffle",
    "batched",
    "unbatched",
    "unlisted",
    "rename",
    "cached",
]


def getfirst(sample, keys, default=None, missing_is_error=True):
    """Look up the first field present among several alternatives."""
    if isinstance(keys, str):
        assert " " not in keys
        keys = keys.split(";")
    for key in keys:
        if key in sample:
            return sample[key]
    if missing_is_error:
        raise ValueError(f"didn't find {keys} in {list(sample.keys())}")
    return default


def _combine(column, combine_tensors=True, combine_scalars=True):
    """Stack one column of a batch, as the reference implementation does."""
    import numpy as np

    first = column[0]
    if isinstance(first, (int, float)) and combine_scalars:
        return np.array(column)
    if isinstance(first, np.ndarray) and combine_tensors:
        shapes = {x.shape for x in column}
        assert len(shapes) == 1, f"all shapes must be equal in collation, got {shapes}"
        return np.array(column)
    try:  # pragma: no cover - depends on the environment
        import torch

        if isinstance(first, torch.Tensor) and combine_tensors:
            return torch.stack(list(column))
    except ModuleNotFoundError:
        pass
    return list(column)


def default_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Stack a batch of samples or tuples, column by column."""
    if isinstance(samples[0], (list, tuple)):
        width = len(samples[0])
        return tuple(
            _combine([s[i] for s in samples], combine_tensors, combine_scalars) for i in range(width)
        )
    keys = set(samples[0].keys())
    for sample in samples[1:]:
        assert set(sample.keys()) == keys, "keys don't match in different samples"
    return {k: _combine([s[k] for s in samples], combine_tensors, combine_scalars) for k in keys}


def handle(handler, exn):
    """Apply an exception handler, defaulting to re-raising."""
    if handler is None:
        raise exn
    return handler(exn)


def shuffle(source, bufsize, seed=None):
    """The reference buffer shuffle, for when the native one cannot be used."""
    rng = random.Random(seed) if seed is not None else random.Random()
    initial = max(1, (bufsize + 9) // 10)
    buf = []

    def take():
        k = rng.randint(0, len(buf) - 1)
        buf[k], buf[-1] = buf[-1], buf[k]
        return buf.pop()

    for sample in source:
        buf.append(sample)
        if len(buf) >= initial:
            yield take()
        if len(buf) > bufsize:
            yield take()
    while buf:
        yield take()


def batched(source, batchsize, partial, collate):
    """Group into batches, collating them if asked."""
    batch = []
    for sample in source:
        batch.append(sample)
        if len(batch) >= batchsize:
            yield collate(batch) if collate else batch
            batch = []
    if batch and (partial or len(batch) == batchsize):
        yield collate(batch) if collate else batch


def unbatched(source):
    """Split batches back into samples."""
    for batch in source:
        if isinstance(batch, (list, tuple)) and batch and isinstance(batch[0], dict):
            yield from batch
        elif isinstance(batch, dict):
            size = next(iter({len(v) for v in batch.values()}))
            for i in range(size):
                yield {k: v[i] for k, v in batch.items()}
        else:
            for i in range(len(batch[0])):
                yield tuple(column[i] for column in batch)


def unlisted(source):
    """Split lists back into samples."""
    for group in source:
        yield from group


def rename(source, mapping, keep):
    """Rename fields, resolving each source from a ``;``-separated list."""
    consumed = {
        part for spec in mapping.values() for part in (spec.split(";") if isinstance(spec, str) else spec)
    }
    for sample in source:
        if keep:
            out = {k: v for k, v in sample.items() if k not in consumed}
        else:
            out = {k: v for k, v in sample.items() if k.startswith("__")}
        for target, spec in mapping.items():
            out[target] = getfirst(sample, spec, missing_is_error=True)
        yield out


def cached():
    """Build a stage that caches the stream in memory on its first pass."""
    cache: list = []
    filled = [False]

    def apply(source):
        if filled[0]:
            yield from cache
            return
        for sample in source:
            cache.append(sample)
            yield sample
        filled[0] = True

    return apply
