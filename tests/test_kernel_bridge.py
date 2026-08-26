from __future__ import annotations

import sys
import types

import pytest
import torch

from rwkv7_hf import kernel_bridge
from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent, rwkv7_recurrent_reference


def recurrent_inputs(*, requires_grad: bool = False):
    torch.manual_seed(7)
    shape = (2, 3, 2, 4)
    values = [
        torch.randn(shape, dtype=torch.float64, requires_grad=requires_grad)
        for _ in range(6)
    ]
    state = torch.randn(
        2, 2, 4, 4, dtype=torch.float64, requires_grad=requires_grad
    )
    mask = torch.tensor([[True, True, True], [False, True, True]])
    return (*values, state, mask)


@pytest.fixture(autouse=True)
def reset_bridge(monkeypatch):
    monkeypatch.delenv("RWKV7_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "rwkv7_kernels", raising=False)
    kernel_bridge.reset_kernel_discovery_for_tests()
    yield
    kernel_bridge.reset_kernel_discovery_for_tests()


def install_fake_kernel(monkeypatch, *, supported=True, reason="test kernel"):
    calls = {"probe": 0, "run": 0}
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = kernel_bridge.KERNEL_API_VERSION

    def probe(*args):
        calls["probe"] += 1
        decision = supported(*args) if callable(supported) else supported
        return {
            "supported": bool(decision),
            "implementation": "fake-recurrent-v1",
            "reason": reason,
        }

    def run(*args):
        calls["run"] += 1
        return rwkv7_recurrent_reference(*args)

    module.probe_recurrent_v1 = probe
    module.recurrent_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    kernel_bridge.reset_kernel_discovery_for_tests()
    return calls


def test_auto_without_package_uses_reference():
    inputs = recurrent_inputs()
    actual = rwkv7_recurrent(*inputs)
    expected = rwkv7_recurrent_reference(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    route = kernel_bridge.last_backend_route()
    assert route is not None
    assert route["requested"] == "auto"
    assert route["selected"] == "reference"


def test_forced_optimized_without_package_fails_closed():
    with pytest.raises(RuntimeError, match="optimized RWKV7 backend is unavailable"):
        rwkv7_recurrent(*recurrent_inputs(), backend="optimized")


def test_supported_kernel_is_selected_and_matches_reference(monkeypatch):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs()
    actual = rwkv7_recurrent(*inputs)
    expected = rwkv7_recurrent_reference(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert calls == {"probe": 1, "run": 1}
    assert kernel_bridge.last_backend_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-recurrent-v1",
        "reason": "test kernel",
    }


def test_unsupported_auto_falls_back_but_forced_mode_fails(monkeypatch):
    calls = install_fake_kernel(monkeypatch, supported=False, reason="no backward")
    inputs = recurrent_inputs(requires_grad=True)
    actual = rwkv7_recurrent(*inputs)
    expected = rwkv7_recurrent_reference(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    assert calls == {"probe": 1, "run": 0}
    assert kernel_bridge.last_backend_route()["reason"] == "no backward"
    with pytest.raises(RuntimeError, match="no backward"):
        rwkv7_recurrent(*inputs, backend="optimized")


def test_context_override_is_scoped(monkeypatch):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs()
    with kernel_bridge.use_rwkv7_backend("reference"):
        rwkv7_recurrent(*inputs)
        assert kernel_bridge.last_backend_route()["requested"] == "reference"
    rwkv7_recurrent(*inputs)
    assert kernel_bridge.last_backend_route()["selected"] == "optimized"
    assert calls == {"probe": 1, "run": 1}


def test_claimed_backend_runtime_error_is_not_hidden(monkeypatch):
    install_fake_kernel(monkeypatch)
    module = sys.modules["rwkv7_kernels"]

    def broken(*args):
        raise RuntimeError("kernel launch failed")

    module.recurrent_v1 = broken
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        rwkv7_recurrent(*recurrent_inputs())


def test_invalid_backend_name_is_rejected():
    with pytest.raises(ValueError, match="one of auto, reference, optimized"):
        rwkv7_recurrent(*recurrent_inputs(), backend="fastest")


def test_training_hint_never_enters_inference_backend(monkeypatch):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs()
    actual = rwkv7_recurrent(*inputs, training=True)
    expected = rwkv7_recurrent_reference(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    assert calls == {"probe": 0, "run": 0}
    assert kernel_bridge.last_backend_route() == {
        "requested": "auto",
        "selected": "reference",
        "implementation": "torch",
        "reason": "the optimized recurrent protocol v1 is inference-only during training",
    }
    with pytest.raises(RuntimeError, match="inference-only during training"):
        rwkv7_recurrent(*inputs, backend="optimized", training=True)
