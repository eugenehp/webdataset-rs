"""Assorted helpers, as in the reference implementation."""

import os

from . import _native

__all__ = [
    "base_plus_ext",
    "pytorch_worker_info",
    "pytorch_worker_seed",
    "repeatedly",
    "identity",
    "is_iterable",
    "PipelineStage",
]

#: Set from the environment; refuses `pipe:` and `file:` URLs when on.
enforce_security = bool(int(os.environ.get("WDS_SECURE", "0")))


class PipelineStage:
    """Base class for pipeline stages."""

    def invoke(self, *args, **kw):
        raise NotImplementedError


def base_plus_ext(path):
    """Split a path into its basename and its full extension."""
    result = _native.base_plus_ext(path)
    return result if result is not None else (None, None)


def identity(x):
    """Return the argument unchanged."""
    return x


def is_iterable(obj):
    """Whether ``obj`` is iterable, strings and bytes aside."""
    if isinstance(obj, (str, bytes)):
        return False
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def pytorch_worker_info(group=None):
    """Report ``(rank, world_size, worker, num_workers)``."""
    info = _native.worker_info()
    rank, world_size = info["rank"], info["world_size"]
    worker, num_workers = info["worker"], info["num_workers"]
    try:  # pragma: no cover - depends on the environment
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker, num_workers = worker_info.id, worker_info.num_workers
    except ModuleNotFoundError:
        pass
    return rank, world_size, worker, num_workers


def pytorch_worker_seed(group=None):
    """A distinct, deterministic seed per worker and node."""
    rank, _, worker, _ = pytorch_worker_info(group=group)
    return rank * 1000 + worker


def guess_batchsize(batch):
    """Guess a batch size from the first column."""
    return len(batch[0])


def repeatedly(source, nepochs=None, nbatches=None, nsamples=None, batchsize=guess_batchsize):
    """Yield from ``source`` over and over."""
    epoch = batch = total = 0
    while True:
        for sample in source:
            yield sample
            batch += 1
            if nbatches is not None and batch >= nbatches:
                return
            if nsamples is not None:
                total += batchsize(sample)
                if total >= nsamples:
                    return
        epoch += 1
        if nepochs is not None and epoch >= nepochs:
            return
