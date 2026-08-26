from __future__ import annotations

import json

from evaluation.validate_finetune_smoke_runs import RUNS, validate


def write(path, name, value):
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(value) + "\n")


def make_run(path, *, kernels="0.10.0.dev0"):
    write(path, "exit_status.json", {"returncode": 0})
    write(
        path,
        "training_checks.json",
        {"finite_loss": True, "nonzero_gradient": True, "global_step": 1},
    )
    write(path, "adapter_reload.json", {"close": True, "max_abs": 0.0})
    write(path, "changed_parameters.json", ["model.layers.0.attn.r_proj.lora_A"])
    write(path, "environment.json", {"rwkv7_hf": "0.10.0.dev0", "rwkv7_kernels": kernels})
    write(path, "resolved_config.json", {"code_sha": "abc", "source_revision": "abc"})
    write(path, "model_provenance.json", {"resolved_revision": "weights"})
    (path / "metrics.jsonl").write_text(json.dumps({"loss": 1.0}) + "\n")


def test_finetune_smoke_bundle_passes(tmp_path):
    for name in RUNS:
        make_run(tmp_path / name)
    assert validate(tmp_path, expected_step=1)["status"] == "passed"


def test_finetune_smoke_requires_companion_wheel(tmp_path):
    for name in RUNS:
        make_run(tmp_path / name)
    write(tmp_path / "dpo", "environment.json", {"rwkv7_hf": "0.10.0.dev0", "rwkv7_kernels": None})
    report = validate(tmp_path, expected_step=1)
    assert report["status"] == "failed"
    assert "rwkv7-kernels distribution was not installed" in report["runs"]["dpo"]["failures"]
