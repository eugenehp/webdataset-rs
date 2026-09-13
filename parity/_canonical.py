"""A canonical byte form for decoded values, shared by the parity dumpers.

Comparing two implementations means comparing the *data* they produce, not the
objects they happen to hold it in. A NumPy array, a torch tensor and a PIL image
of the same pixels are the same answer; a digest taken over this form says so,
while a digest over a pickle or a repr would not.

The form is deliberately dull and self-describing — a type tag, then the bytes —
so the Rust dumper can reproduce it exactly. Its counterpart lives in
``crates/webdataset-tools/src/bin/parity-dump.rs``; the two must agree, which is what
the parity runs check.
"""

import hashlib

import numpy as np

__all__ = ["canonical", "describe", "digest", "as_array"]


def as_array(value):
    """Reduce a tensor-like object to the array it wraps.

    Torch tensors and PIL images are compared as the arrays they hold, since
    that is the data; which library is holding it is not.
    """
    module = type(value).__module__
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    if module.startswith("PIL"):
        return np.asarray(value)
    return value


def canonical(value):
    """Reduce a decoded value to bytes any correct implementation would produce."""
    value = as_array(value)

    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"text:" + value.encode("utf-8")
    # `bool` is checked before `int`, since in Python it is one.
    if isinstance(value, (bool, np.bool_)):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, (int, np.integer)):
        return b"int:" + str(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        return b"float:" + repr(float(value)).encode("ascii")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = f"tensor:{array.dtype.name}:{list(array.shape)}:".encode("ascii")
        return header + array.tobytes()
    if isinstance(value, (list, tuple)):
        return b"list:[" + b",".join(canonical(v) for v in value) + b"]"
    if isinstance(value, dict):
        parts = [k.encode("utf-8") + b"=" + canonical(v) for k, v in sorted(value.items())]
        return b"map:{" + b",".join(parts) + b"}"
    if value is None:
        return b"null"

    raise TypeError(f"no canonical form for {type(value).__name__}")


def describe(value):
    """A short type tag, so a mismatch reads clearly."""
    value = as_array(value)

    if isinstance(value, np.ndarray):
        return f"tensor<{value.dtype.name}>{list(value.shape)}"
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    if isinstance(value, bytes):
        return f"bytes[{len(value)}]"
    # Plain `list` and `dict`, without a length: these tags are compared against
    # the ones the Rust dumper emits, and a length there would say nothing the
    # digest does not already cover.
    if isinstance(value, tuple):
        return "list"
    return type(value).__name__


def digest(value):
    """The SHA-256 of a value's canonical form."""
    return hashlib.sha256(canonical(value)).hexdigest()
