#!/usr/bin/env python3
"""Render sorted raw Qwen Prefill/Decode results for the optimized HF lane."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        PAIR_RANK,
        read_rows,
        validate_matrix,
    )
except ModuleNotFoundError:
    from validate_qwen35_best_optimized_hf_v1 import (
        PAIR_RANK,
        read_rows,
        validate_matrix,
    )


def display_rate(value: float) -> str:
    """Apply the report-only threshold without modifying machine-readable data."""

    return f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            PAIR_RANK.get(str(row.get("model_pair")), 999),
            str(row.get("device", "")),
            int(row.get("batch_size", 0)),
            int(row.get("prompt_tokens", 0)),
            int(row.get("decode_tokens", 0)),
        ),
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = ordered_rows(rows)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in ordered:
        key = (
            str(row["model_size_label"]),
            str(row["device"]),
            int(row["batch_size"]),
        )
        groups.setdefault(key, []).append(row)
    medians = [
        {
            "model_size_label": model,
            "device": device,
            "batch_size": batch,
            "cells": len(group),
            "prefill_tokps_median": statistics.median(
                float(row["prefill_tokps_total"]) for row in group
            ),
            "decode_tokps_median": statistics.median(
                float(row["decode_tokps_total"]) for row in group
            ),
        }
        for (model, device, batch), group in groups.items()
    ]
    return {
        "schema_version": 1,
        "benchmark_matrix": "qwen35_best_optimized_hf_v1",
        "sort_order": ["model_size", "gpu", "batch", "prompt", "decode"],
        "display_rounding": {">=100": 0, "<100": 1},
        "rows": len(ordered),
        "model_batch_medians": medians,
        "cells": [
            {
                "model_pair": row["model_pair"],
                "model_size_label": row["model_size_label"],
                "device": row["device"],
                "batch_size": row["batch_size"],
                "prompt_tokens": row["prompt_tokens"],
                "decode_tokens": row["decode_tokens"],
                "prefill_tokps_total": row["prefill_tokps_total"],
                "decode_tokps_total": row["decode_tokps_total"],
                "prefill_tokps_total_raw": row["prefill_tokps_total_raw"],
                "decode_tokps_total_raw": row["decode_tokps_total_raw"],
                "prefill_sec_samples": row["prefill_sec_samples"],
                "decode_sec_samples": row["decode_sec_samples"],
            }
            for row in ordered
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RTX 5090 Qwen3.5 best-optimized HF raw results",
        "",
        "Rows are sorted by model size, GPU, B1/B8, prompt and decode. "
        "Display values use 0 decimals at >=100 tok/s and 1 decimal below 100; "
        "JSONL and JSON retain the original numeric values and all seven samples.",
        "",
        "## Model / batch medians",
        "",
        "| Qwen3.5 | GPU | Batch | Cells | Prefill tok/s | Decode tok/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["model_batch_medians"]:
        lines.append(
            "| {model_size_label} | {device} | B{batch_size} | {cells} | {prefill} | {decode} |".format(
                **row,
                prefill=display_rate(float(row["prefill_tokps_median"])),
                decode=display_rate(float(row["decode_tokps_median"])),
            )
        )
    lines.extend(
        [
            "",
            "## Complete 48-cell raw matrix",
            "",
            "| Qwen3.5 | GPU | Batch | Prompt | Decode | Prefill tok/s | Decode tok/s |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cells"]:
        lines.append(
            "| {model_size_label} | {device} | B{batch_size} | {prompt_tokens} | "
            "{decode_tokens} | {prefill} | {decode} |".format(
                **row,
                prefill=display_rate(float(row["prefill_tokps_total"])),
                decode=display_rate(float(row["decode_tokps_total"])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--expected-device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.results)
    validation = validate_matrix(rows, expected_device=args.expected_device)
    if validation["status"] != "pass":
        print(
            "QWEN35_BEST_OPTIMIZED_SUMMARY "
            + json.dumps(
                {"status": "fail", "errors": validation["errors"]},
                ensure_ascii=False,
            )
        )
        return 1
    summary = build_summary(rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(
        "QWEN35_BEST_OPTIMIZED_SUMMARY "
        + json.dumps({"status": "pass", "rows": summary["rows"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
