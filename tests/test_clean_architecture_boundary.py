from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(name: str) -> str:
    return (ROOT / "rwkv7_hf" / name).read_text(encoding="utf-8")


def test_modeling_keeps_full_public_architecture_and_no_hardware_policy():
    text = source("modeling_rwkv7.py")
    tree = ast.parse(text)
    classes = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    assert {
        "RWKV7TimeMix",
        "RWKV7ChannelMix",
        "RWKV7Block",
        "RWKV7PreTrainedModel",
        "RWKV7Model",
        "RWKV7ForCausalLM",
    } <= classes
    for forbidden in (
        "torch.cuda",
        "CUDAGraph",
        "rwkv7_kernels",
        "RWKV7_BACKEND",
        "triton",
        "device_capability",
    ):
        assert forbidden not in text


def test_config_and_reference_operator_have_no_backend_policy():
    config = source("configuration_rwkv7.py")
    operator = source("ops_rwkv7.py")
    for forbidden in (
        "RWKV7_BACKEND",
        "rwkv7_kernels",
        "CUDAGraph",
        "torch.cuda",
        "triton",
    ):
        assert forbidden not in config
        assert forbidden not in operator
    assert "def rwkv7_recurrent_reference(" in operator
    assert "try_optimized_recurrent(" in operator


def test_base_distribution_does_not_depend_on_companion_wheel():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "rwkv7-kernels" not in project
    assert "rwkv7_kernels" not in project
