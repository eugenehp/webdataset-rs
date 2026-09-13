"""Pluggable exception handlers, as in the reference implementation.

A handler takes an exception and returns ``True`` to skip the offending item,
``False`` to stop the stream, or raises to report it.
"""

import time
import warnings

__all__ = [
    "reraise_exception",
    "ignore_and_continue",
    "warn_and_continue",
    "ignore_and_stop",
    "warn_and_stop",
]


def reraise_exception(exn):
    """Re-raise the exception."""
    raise exn


def ignore_and_continue(exn):
    """Drop the offending item and carry on."""
    return True


def warn_and_continue(exn):
    """Warn, drop the offending item, and carry on."""
    warnings.warn(repr(exn), stacklevel=2)
    # Slow the stream a little so warnings do not scroll past unnoticed.
    time.sleep(0.5)
    return True


def ignore_and_stop(exn):
    """End the stream."""
    return False


def warn_and_stop(exn):
    """Warn and end the stream."""
    warnings.warn(repr(exn), stacklevel=2)
    time.sleep(0.5)
    return False
