#!/usr/bin/env python3
"""Compare an optimized lm_eval matrix with the readable reference run.

Both runs must contain the same formal units, dataset fingerprints, metric
names, and finite metric values.  This lets an optional backend be exercised
with ``RWKV7_BACKEND=optimized`` without weakening the normal HF gate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from .common import task_dataset_fingerprint
except ImportError:  # direct script execution
    from common import task_dataset_fingerprint


IGNORED_JSON_NAMES = {
    "environment.json",
    "manifest.json",
    "merge.json",
    "models.json",
    "validation.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference and optional-backend lm_eval matrices"
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=48)
    parser.add_argument("--metric-atol", type=float, default=0.0)
    parser.add_argument("--metric-rtol", type=float, default=0.0)
    return parser.parse_args()


def read_manifest(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        unit = str(row["unit"])
        if unit in rows and rows[unit] != row:
            raise ValueError(f"conflicting duplicate manifest unit: {unit}")
        rows[unit] = row
    return rows


def find_result(root: Path, unit: str) -> Path:
    directory = root / unit
    candidates = [
        path
        for path in directory.rglob("*.json")
        if path.name not in IGNORED_JSON_NAMES
        and not path.name.startswith("samples_")
    ]
    if not candidates:
        raise FileNotFoundError(f"no lm_eval result JSON for {unit} in {directory}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def numeric_metrics(payload: dict[str, Any], task: str) -> dict[str, float]:
    values = payload["results"][task]
    return {
        key: float(value)
        for key, value in values.items()
        if isinstance(value, (int, float))
        and not key.split(",", 1)[0].endswith("_stderr")
    }


def logical_model(row: dict[str, Any]) -> str:
    label = row.get("model_label")
    if label is not None:
        return str(label)
    return str(row["unit"]).split("-b", 1)[0]


def normalized_dataset_fingerprint(row: dict[str, Any]) -> str | None:
    provenance = row.get("task_provenance", {})
    config = provenance.get("task_config")
    sample_count = provenance.get("sample_count")
    sample_hash = provenance.get("sample_hash_fingerprint")
    if not isinstance(config, dict) or sample_count is None or not sample_hash:
        return provenance.get("dataset_fingerprint")
    return task_dataset_fingerprint(config, int(sample_count), str(sample_hash))


def compare(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    expected_units: int,
    metric_atol: float,
    metric_rtol: float,
) -> dict[str, Any]:
    reference = read_manifest(reference_dir)
    candidate = read_manifest(candidate_dir)
    reference_units = set(reference)
    candidate_units = set(candidate)
    failures: list[dict[str, Any]] = []

    if len(reference) != expected_units or len(candidate) != expected_units:
        failures.append(
            {
                "kind": "unit_count",
                "expected": expected_units,
                "reference": len(reference),
                "candidate": len(candidate),
            }
        )
    for unit in sorted(reference_units - candidate_units):
        failures.append({"kind": "missing_candidate_unit", "unit": unit})
    for unit in sorted(candidate_units - reference_units):
        failures.append({"kind": "unexpected_candidate_unit", "unit": unit})

    compared_metrics = 0
    for unit in sorted(reference_units & candidate_units):
        left_row = reference[unit]
        right_row = candidate[unit]
        for side, row in (("reference", left_row), ("candidate", right_row)):
            if int(row.get("exit_code", 1)) != 0:
                failures.append({"kind": "nonzero_exit", "unit": unit, "side": side})
            if not row.get("formal"):
                failures.append({"kind": "nonformal_unit", "unit": unit, "side": side})
        if logical_model(left_row) != logical_model(right_row):
            failures.append(
                {
                    "kind": "manifest_mismatch",
                    "unit": unit,
                    "field": "model_label",
                    "reference": logical_model(left_row),
                    "candidate": logical_model(right_row),
                }
            )
        for key in ("task", "batch_size"):
            if left_row.get(key) != right_row.get(key):
                failures.append(
                    {
                        "kind": "manifest_mismatch",
                        "unit": unit,
                        "field": key,
                        "reference": left_row.get(key),
                        "candidate": right_row.get(key),
                    }
                )
        left_fingerprint = normalized_dataset_fingerprint(left_row)
        right_fingerprint = normalized_dataset_fingerprint(right_row)
        if not left_fingerprint or left_fingerprint != right_fingerprint:
            failures.append(
                {
                    "kind": "dataset_fingerprint_mismatch",
                    "unit": unit,
                    "reference": left_fingerprint,
                    "candidate": right_fingerprint,
                }
            )

        task = str(left_row["task"])
        left_payload = json.loads(find_result(reference_dir, unit).read_text())
        right_payload = json.loads(find_result(candidate_dir, unit).read_text())
        left_metrics = numeric_metrics(left_payload, task)
        right_metrics = numeric_metrics(right_payload, task)
        if set(left_metrics) != set(right_metrics):
            failures.append(
                {
                    "kind": "metric_set_mismatch",
                    "unit": unit,
                    "reference": sorted(left_metrics),
                    "candidate": sorted(right_metrics),
                }
            )
        for name in sorted(set(left_metrics) & set(right_metrics)):
            expected = left_metrics[name]
            actual = right_metrics[name]
            compared_metrics += 1
            finite = math.isfinite(expected) and math.isfinite(actual)
            tolerance = metric_atol + metric_rtol * abs(expected)
            absolute = abs(actual - expected) if finite else math.inf
            if not finite or absolute > tolerance:
                failures.append(
                    {
                        "kind": "metric_mismatch",
                        "unit": unit,
                        "metric": name,
                        "reference": expected,
                        "candidate": actual,
                        "absolute": absolute,
                        "tolerance": tolerance,
                    }
                )

    return {
        "schema": "rwkv7-lm-eval-backend-parity-v1",
        "status": "passed" if not failures else "failed",
        "expected_units": expected_units,
        "reference_units": len(reference),
        "candidate_units": len(candidate),
        "compared_metrics": compared_metrics,
        "metric_atol": metric_atol,
        "metric_rtol": metric_rtol,
        "reference_code_shas": sorted(
            {str(row.get("code_sha")) for row in reference.values()}
        ),
        "candidate_code_shas": sorted(
            {str(row.get("code_sha")) for row in candidate.values()}
        ),
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    if args.expected_units < 1:
        raise SystemExit("--expected-units must be positive")
    if args.metric_atol < 0 or args.metric_rtol < 0:
        raise SystemExit("metric tolerances must be non-negative")
    report = compare(
        args.reference_dir.resolve(),
        args.candidate_dir.resolve(),
        expected_units=args.expected_units,
        metric_atol=args.metric_atol,
        metric_rtol=args.metric_rtol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
