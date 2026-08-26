from __future__ import annotations

from pathlib import Path
import sys

import torch

from rwkv7_hf import kernel_bridge
from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent, rwkv7_recurrent_reference


def test_source_kernel_package_exposes_protocol_and_cpu_falls_back(monkeypatch):
    kernel_root = Path(__file__).resolve().parents[1] / "kernel_wheel"
    monkeypatch.syspath_prepend(str(kernel_root))
    monkeypatch.delitem(sys.modules, "rwkv7_kernels", raising=False)
    kernel_bridge.reset_kernel_discovery_for_tests()

    import rwkv7_kernels

    assert rwkv7_kernels.RWKV7_KERNEL_API_VERSION == 1
    shape = (1, 2, 1, 64)
    values = [torch.randn(shape) for _ in range(6)]
    state = torch.randn(1, 1, 64, 64)
    support = rwkv7_kernels.probe_recurrent_v1(*values, state, None)
    assert not support["supported"]
    assert any(word in support["reason"] for word in ("CUDA", "Triton"))

    actual = rwkv7_recurrent(*values, state)
    expected = rwkv7_recurrent_reference(*values, state)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert kernel_bridge.last_backend_route()["selected"] == "reference"
