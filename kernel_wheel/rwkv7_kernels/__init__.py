"""Versioned optional operator protocol consumed by :mod:`rwkv7_hf`."""
from __future__ import annotations

from .recurrent_graph import probe_recurrent_v1, recurrent_v1


RWKV7_KERNEL_API_VERSION = 1


__all__ = [
    "RWKV7_KERNEL_API_VERSION",
    "probe_recurrent_v1",
    "recurrent_v1",
]
