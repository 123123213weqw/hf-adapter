"""Versioned optional operator protocol consumed by :mod:`rwkv7_hf`."""
from __future__ import annotations

import os

from .recurrent_graph import (
    probe_recurrent_v1 as _probe_graph,
    recurrent_v1 as _run_graph,
)
from .recurrent_triton import (
    probe_recurrent_v1 as _probe_triton,
    recurrent_v1 as _run_triton,
)


RWKV7_KERNEL_API_VERSION = 1


def _implementation():
    name = os.environ.get("RWKV7_KERNEL_IMPL", "graph").strip().lower()
    if name == "graph":
        return _probe_graph, _run_graph
    if name == "triton":
        return _probe_triton, _run_triton
    if name == "auto":
        return None
    raise ValueError(
        "RWKV7_KERNEL_IMPL must be one of auto, graph, or triton; "
        f"got {name!r}"
    )


def probe_recurrent_v1(*args, **kwargs):
    selected = _implementation()
    if selected is not None:
        return selected[0](*args, **kwargs)
    triton_support = _probe_triton(*args, **kwargs)
    if triton_support.get("supported"):
        return triton_support
    return _probe_graph(*args, **kwargs)


def recurrent_v1(*args, **kwargs):
    selected = _implementation()
    if selected is not None:
        return selected[1](*args, **kwargs)
    triton_support = _probe_triton(*args, **kwargs)
    if triton_support.get("supported"):
        return _run_triton(*args, **kwargs)
    return _run_graph(*args, **kwargs)


__all__ = [
    "RWKV7_KERNEL_API_VERSION",
    "probe_recurrent_v1",
    "recurrent_v1",
]
