"""The tenbin (``.ten``) binary tensor format, backed by Rust."""

import sys

from . import _native

__all__ = ["encode_buffer", "decode_buffer", "save", "load", "write", "read"]


def encode_buffer(arrays, infos=None):
    """Encode a list of arrays into a tenbin byte string."""
    if not isinstance(arrays, list):
        raise ValueError("requires a list")
    return _native.tenbin_encode(arrays)


def decode_buffer(data, infos=False):
    """Decode a tenbin byte string into a list of arrays."""
    return _native.tenbin_decode(bytes(data))


def write(stream, arrays, infos=None):
    """Write arrays to a stream."""
    stream.write(encode_buffer(list(arrays)))


def read(stream, n=sys.maxsize, infos=False):
    """Read arrays from a stream."""
    return decode_buffer(stream.read())


def save(fname, *args, infos=None, nocheck=False):
    """Write arrays to a ``.ten`` file."""
    if not nocheck and not fname.endswith(".ten"):
        raise ValueError("file name should end in .ten")
    with open(fname, "wb") as stream:
        write(stream, list(args), infos=infos)


def load(fname, infos=False, nocheck=False):
    """Read arrays from a ``.ten`` file."""
    if not nocheck and not fname.endswith(".ten"):
        raise ValueError("file name should end in .ten")
    with open(fname, "rb") as stream:
        return read(stream, infos=infos)
