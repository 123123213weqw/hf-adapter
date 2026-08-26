from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_validator():
    path = Path(__file__).parents[1] / "evaluation" / "validate_lm_eval_matrix.py"
    spec = importlib.util.spec_from_file_location("validate_lm_eval_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_mismatch_is_reported_once(monkeypatch, tmp_path: Path) -> None:
    validator = _load_validator()
    manifest = []
    for model_index in range(3):
        for task_index in range(8):
            task = f"task_{task_index}"
            for batch_size in (1, 8):
                unit = f"model_{model_index}-{task}-b{batch_size}"
                unit_dir = tmp_path / unit
                unit_dir.mkdir()
                value = 0.002 if (model_index, task_index, batch_size) == (0, 0, 8) else 0.0
                (unit_dir / "results.json").write_text(
                    json.dumps({"results": {task: {"acc": value}}})
                )
                manifest.append(
                    {
                        "unit": unit,
                        "exit_code": 0,
                        "formal": True,
                        "code_sha": "deadbeef",
                        "task_provenance": {"dataset_fingerprint": "fixture"},
                        "model": f"model_{model_index}",
                        "task": task,
                        "batch_size": batch_size,
                    }
                )
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    monkeypatch.setattr(sys, "argv", ["validate", "--result-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        validator.main()

    assert exc_info.value.code == 1
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["status"] == "failed"
    assert len(report["failures"]) == 1
