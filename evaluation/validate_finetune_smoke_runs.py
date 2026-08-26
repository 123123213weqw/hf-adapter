#!/usr/bin/env python3
"""Validate short SFT/DPO/GRPO runs used by an optional-backend gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


RUNS = ("sft", "dpo", "grpo")


def read(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_one(path: Path, *, expected_step: int) -> dict[str, Any]:
    failures: list[str] = []
    try:
        exit_status = read(path / "exit_status.json")
        checks = read(path / "training_checks.json")
        reload = read(path / "adapter_reload.json")
        changed = read(path / "changed_parameters.json")
        environment = read(path / "environment.json")
        config = read(path / "resolved_config.json")
        model = read(path / "model_provenance.json")
        metrics = [
            json.loads(line)
            for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {"path": str(path), "status": "failed", "failures": [str(error)]}

    if exit_status.get("returncode") != 0:
        failures.append("nonzero exit")
    if not checks.get("finite_loss"):
        failures.append("loss was not finite")
    if not checks.get("nonzero_gradient"):
        failures.append("no nonzero gradient was observed")
    if int(checks.get("global_step", -1)) != expected_step:
        failures.append(f"global_step != {expected_step}")
    if not reload.get("close"):
        failures.append("adapter save/reload logits changed")
    if not changed:
        failures.append("no trainable parameter changed")
    numeric = [
        float(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    if not metrics or any(not math.isfinite(value) for value in numeric):
        failures.append("metrics are missing or non-finite")
    if not config.get("code_sha") or not config.get("source_revision"):
        failures.append("source provenance is missing")
    if not model.get("resolved_revision"):
        failures.append("model provenance is missing")
    if not environment.get("rwkv7_hf"):
        failures.append("rwkv7-hf distribution was not installed")
    if not environment.get("rwkv7_kernels"):
        failures.append("rwkv7-kernels distribution was not installed")
    return {
        "path": str(path),
        "status": "passed" if not failures else "failed",
        "global_step": checks.get("global_step"),
        "rwkv7_hf": environment.get("rwkv7_hf"),
        "rwkv7_kernels": environment.get("rwkv7_kernels"),
        "changed_parameters": len(changed),
        "adapter_reload_max_abs": reload.get("max_abs"),
        "failures": failures,
    }


def validate(root: Path, *, expected_step: int) -> dict[str, Any]:
    runs = {
        name: validate_one(root / name, expected_step=expected_step) for name in RUNS
    }
    passed = all(row["status"] == "passed" for row in runs.values())
    return {
        "schema": "rwkv7-native-backend-finetune-smoke-v1",
        "status": "passed" if passed else "failed",
        "expected_step": expected_step,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=1)
    args = parser.parse_args()
    report = validate(args.result_dir.resolve(), expected_step=args.expected_step)
    output = args.result_dir / "validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
