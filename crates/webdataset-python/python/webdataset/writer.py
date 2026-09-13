"""Writing shards, backed by Rust."""

import io

from . import _native

__all__ = ["TarWriter", "ShardWriter", "numpy_dumps", "torch_dumps"]

TarWriter = _native.TarWriter
ShardWriter = _native.ShardWriter


def numpy_dumps(data):
    """Encode an array in NumPy's ``.npy`` format."""
    import numpy.lib.format

    stream = io.BytesIO()
    numpy.lib.format.write_array(stream, data)
    return stream.getvalue()


def torch_dumps(data):
    """Encode an object with ``torch.save``."""
    import torch

    stream = io.BytesIO()
    torch.save(data, stream)
    return stream.getvalue()
