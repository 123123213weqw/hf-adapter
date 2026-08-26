from __future__ import annotations

import json

from evaluation.compare_lm_eval_matrices import compare


def write_matrix(
    root,
    *,
    metric=0.5,
    sample_hash="same-samples",
    model_path="/host-a/model",
):
    unit = "0.1b-b1-piqa"
    result_dir = root / unit / "run"
    result_dir.mkdir(parents=True)
    (result_dir / "results_test.json").write_text(
        json.dumps(
            {
                "results": {
                    "piqa": {
                        "acc,none": metric,
                        "acc_stderr,none": 0.123,
                    }
                }
            }
        )
    )
    row = {
        "unit": unit,
        "model_label": "0.1b",
        "model": model_path,
        "task": "piqa",
        "batch_size": 1,
        "exit_code": 0,
        "formal": True,
        "code_sha": "test-sha",
        "task_provenance": {
            "dataset_fingerprint": "legacy-host-dependent-value",
            "task_config": {
                "task": "piqa",
                "metadata": {"version": 1.0, "pretrained": model_path},
            },
            "sample_count": 2,
            "sample_hash_fingerprint": sample_hash,
        },
    }
    (root / "manifest.jsonl").write_text(json.dumps(row) + "\n")


def test_identical_lm_eval_matrices_pass(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_matrix(reference)
    write_matrix(candidate, model_path="/host-b/same-model")
    report = compare(
        reference,
        candidate,
        expected_units=1,
        metric_atol=0.0,
        metric_rtol=0.0,
    )
    assert report["status"] == "passed"
    assert report["compared_metrics"] == 1


def test_metric_or_dataset_drift_fails(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    write_matrix(reference)
    write_matrix(candidate, metric=0.6, sample_hash="changed")
    report = compare(
        reference,
        candidate,
        expected_units=1,
        metric_atol=0.0,
        metric_rtol=0.0,
    )
    assert report["status"] == "failed"
    assert {failure["kind"] for failure in report["failures"]} == {
        "dataset_fingerprint_mismatch",
        "metric_mismatch",
    }
