from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


_RUNTIME_TASK_METADATA = {
    "backend",
    "device",
    "dtype",
    "max_length",
    "model",
    "model_args",
    "pretrained",
    "revision",
    "trust_remote_code",
}


def canonical_task_config(config: dict) -> dict:
    """Remove model-runtime fields from an lm_eval dataset identity.

    ``lm_eval`` writes HFLM arguments such as the absolute checkpoint path into
    ``task_config.metadata``. They are useful execution provenance, but they do
    not describe the task or its examples and must not make two hosts look like
    different datasets.
    """

    canonical = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    metadata = canonical.get("metadata")
    if isinstance(metadata, dict):
        for key in _RUNTIME_TASK_METADATA:
            metadata.pop(key, None)
        if not metadata:
            canonical.pop("metadata", None)
    return canonical


def task_dataset_fingerprint(
    task_config: dict,
    sample_count: int,
    sample_hash_fingerprint: str,
) -> str:
    canonical = json.dumps(
        {
            "task_config": canonical_task_config(task_config),
            "sample_count": int(sample_count),
            "sample_hash_fingerprint": str(sample_hash_fingerprint),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(model_dir: Path) -> dict:
    files = []
    for path in sorted(model_dir.glob("*.safetensors")):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    config = model_dir / "config.json"
    return {
        "path": str(model_dir.resolve()),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "weights": files,
    }


def environment() -> dict:
    import torch
    import transformers

    def package(name: str):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()[0]
    except Exception:
        driver = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "flash_linear_attention": package("flash-linear-attention"),
        "triton": package("triton"),
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_bundle(output_dir: Path, name: str, report: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    jsonl_path = output_dir / f"{name}.jsonl"
    markdown_path = output_dir / f"{name}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    jsonl_path.write_text(
        json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {name}",
        "",
        f"- status: **{report.get('status', 'unknown')}**",
        f"- model: {report.get('model', {}).get('path')}",
        f"- dtype: {report.get('dtype')}",
        f"- device: {report.get('environment', {}).get('gpu')}",
        f"- code: {report.get('code_sha')}",
    ]
    if report.get("fla_commit") is not None:
        lines.append(f"- FLA: {report.get('fla_commit')}")
    lines.extend([
        "",
        "| case | cosine | max abs | mean abs | argmax |",
        "|---|---:|---:|---:|---|",
    ])
    for case, row in report.get("comparisons", {}).items():
        lines.append(
            f"| {case} | {row['cosine']:.8f} | {row['max_abs']:.8f} | "
            f"{row['mean_abs']:.8f} | {row['argmax_same']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, markdown_path
