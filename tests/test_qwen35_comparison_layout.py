from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = {
    "0.4B / 0.8B": 0,
    "1.5B / 2B": 1,
    "2.9B / 4B": 2,
    "7.2B / 9B": 3,
}
GPU_ORDER = {
    "V100 32GB": 0,
    "RTX 3090": 1,
    "RTX 4080": 2,
    "RTX 4090": 3,
    "RTX 5070 Laptop": 4,
    "RTX 5090": 5,
}


def comparison_rows(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("| GPU |") and "RWKV P / D tok/s" in line
    )
    rows: list[list[str]] = []
    for line in lines[header + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def assert_throughput_format(cell: str) -> None:
    values = cell.replace("**", "").split("/")
    assert len(values) == 2
    for value in values:
        rendered = value.strip()
        numeric = float(rendered.replace(",", ""))
        if numeric >= 100:
            assert "." not in rendered, rendered
        else:
            assert re.fullmatch(r"\d{1,2}\.\d", rendered), rendered


def assert_document_throughput_format(text: str) -> None:
    values = re.findall(r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*tok/s", text)
    pair_values = [
        value
        for pair in re.findall(r"\*\*([0-9,.]+) / ([0-9,.]+)\*\*", text)
        for value in pair
    ]
    for value in (*values, *pair_values):
        assert_throughput_format(f"{value} / {value}")


def test_cross_card_table_is_model_gpu_batch_sorted_and_formatted() -> None:
    for relative in (
        "docs/QWEN35_SPEED_COMPARISON.md",
        "docs/QWEN35_SPEED_COMPARISON_ZH.md",
    ):
        rows = comparison_rows(ROOT / relative)
        assert_document_throughput_format((ROOT / relative).read_text(encoding="utf-8"))
        keys = [
            (MODEL_ORDER[row[1]], GPU_ORDER[row[0]], int(row[2].removeprefix("B")))
            for row in rows
        ]
        assert keys == sorted(keys), relative
        for row in rows:
            assert_throughput_format(row[6])
            assert_throughput_format(row[7])


def test_4090_artifact_table_is_model_batch_sorted_and_formatted() -> None:
    path = ROOT / "bench/4090_hf_best_optimized_v1_20260812/README.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines.index(
        "| RWKV / Qwen | Batch | RWKV Prefill / Decode | Qwen Prefill / Decode | Raw Prefill / Decode | Adjusted Prefill / Decode | Adjusted minima P / D |"
    )
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines[header + 2 : header + 10]
    ]
    keys = [(MODEL_ORDER[row[0]], int(row[1].removeprefix("B"))) for row in rows]
    assert keys == sorted(keys)
    for row in rows:
        assert_throughput_format(row[2])
        assert_throughput_format(row[3])
