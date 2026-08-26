from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_pool_module():
    path = Path(__file__).parents[1] / "evaluation" / "run_lm_eval_v100_pool.py"
    spec = importlib.util.spec_from_file_location("run_lm_eval_v100_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_pool_forwards_wandb_args(monkeypatch, tmp_path: Path) -> None:
    pool = _load_pool_module()
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="ok\n")

    monkeypatch.setattr(pool.subprocess, "run", fake_run)
    model_root = tmp_path / "models"
    (model_root / "rwkv7_01b_hf").mkdir(parents=True)

    result = pool.run_unit(
        repo_root=tmp_path,
        output_dir=tmp_path / "results",
        model_root=model_root,
        python="python",
        code_sha="deadbeef",
        gpu="0",
        label="0.1b",
        folder="rwkv7_01b_hf",
        task="piqa",
        batch_size=1,
        wandb_args="project=rwkv7-hf,group=optimized",
        force=False,
    )

    assert result["exit_code"] == 0
    assert commands[0][-2:] == [
        "--wandb-args",
        "project=rwkv7-hf,group=optimized",
    ]
