from __future__ import annotations

from itertools import product

from bench.validate_qwen35_best_optimized_hf_v1 import PAIRS, validate_matrix
from bench.summarize_qwen35_best_optimized_hf_v1 import (
    build_summary,
    display_rate,
    ordered_rows,
)


def row(pair: str, batch: int, prompt: int, decode: int) -> dict:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_best_optimized_hf_v1",
        "optimization_lane": "qwen_best_optimized_hf",
        "model_pair": pair,
        "model_size_label": PAIRS[pair],
        "model_role": "reference",
        "model_kind": "qwen35",
        "dtype": "fp16",
        "quantization": "none",
        "batch_size": batch,
        "prompt_tokens": prompt,
        "decode_tokens": decode,
        "prefill_chunk_size": 512,
        "warmup": 3,
        "runs": 7,
        "timing_statistic": "median",
        "mtp_enabled": False,
        "speculative_decoding_enabled": False,
        "resident_sweep": True,
        "status": "pass",
        "device": "NVIDIA GeForce RTX 5090",
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
        "triton_version": "3.4.0",
        "transformers_version": "5.12.1",
        "fla_version": "0.5.1",
        "causal_conv1d_version": "1.6.2.post1",
        "qwen_backend_requested": "fla",
        "qwen_conv_backend_requested": "causal_conv1d",
        "qwen_fast_path_required": True,
        "qwen_fast_path_available": True,
        "qwen_fast_path_verified": True,
        "qwen_full_fused_contract_pass": True,
        "qwen_causal_conv1d_importable": True,
        "qwen_conv_backend_effective": "causal_conv1d",
        "qwen_force_torch": False,
        "qwen_decode_optimization_requested": "static_cache_inductor_cudagraph",
        "qwen_decode_optimization_effective": "static_cache_inductor_cudagraph",
        "step_backend": "qwen_static_cache_inductor_cudagraph",
        "prefill_backend_effective": "module_call_dynamic_cache",
        "prefill_cache_type": "DynamicCache",
        "cache_type": "StaticCache",
        "qwen_compile_backend_effective": "inductor",
        "qwen_compile_mode_effective": "reduce-overhead",
        "qwen_compile_fullgraph_effective": False,
        "qwen_compile_dynamic_effective": False,
        "qwen_cuda_graph_requested": True,
        "qwen_cuda_graph_effective": True,
        "qwen_decode_cuda_graph_verified": True,
        "qwen_graph_break_count": 0,
        "qwen_cudagraph_skip_count": 0,
        "qwen_cudagraph_recorded_non_static_inputs": 1,
        "qwen_cuda_graph_launch_count": 1,
        "qwen_cache_pointer_stable": True,
        "qwen_cache_tensor_pointer_count": 54,
        "qwen_graph_parity_verified": True,
        "qwen_graph_prefill_next_token_match": True,
        "qwen_axis_composition": "independent_best_prefill_and_decode",
        "qwen_graph_greedy_match": True,
        "qwen_static_cache_eager_greedy_match": True,
        "qwen_graph_logits_greedy_match": True,
        "qwen_graph_logits_min_cosine": 0.99999,
        "qwen_graph_max_cache_len": prompt + 3 + decode,
        "qwen_graph_probe_tokens": 3 + decode,
        "qwen_graph_logits_probe_tokens": 16,
        "qwen_graph_distinct_batch_prompts": batch > 1,
        "prefill_sec_samples": [0.1] * 7,
        "prefill_sec_median": 0.1,
        "prefill_sec_median_raw": 0.1,
        "decode_sec_samples": [0.2] * 7,
        "decode_sec_median": 0.2,
        "decode_sec_median_raw": 0.2,
        "prefill_tokps_total": 1000.0,
        "prefill_tokps_total_raw": 1000.0,
        "decode_tokps_total": 500.0,
        "decode_tokps_total_raw": 500.0,
        "logits_finite": True,
    }


def complete_rows() -> list[dict]:
    return [
        row(pair, batch, prompt, decode)
        for pair, (batch, prompt, decode) in product(
            PAIRS, product((1, 8), (128, 512, 2048), (128, 512))
        )
    ]


def test_complete_best_optimized_qwen_matrix_passes() -> None:
    summary = validate_matrix(
        complete_rows(), expected_device="NVIDIA GeForce RTX 5090"
    )
    assert summary["status"] == "pass"
    assert summary["reference_rows"] == 48
    assert summary["reference_lane_eligible"] is True
    assert summary["unified_main_table_eligible"] is False


def test_graph_fallback_or_missing_cell_fails() -> None:
    rows = complete_rows()
    rows[0]["qwen_decode_cuda_graph_verified"] = False
    summary = validate_matrix(rows[:-1])
    assert summary["status"] == "fail"
    assert any("qwen_decode_cuda_graph_verified=False" in error for error in summary["errors"])
    assert any("missing cells" in error for error in summary["errors"])


def test_summary_sort_and_display_rounding_do_not_change_raw_values() -> None:
    rows = complete_rows()
    rows.reverse()
    rows[0]["decode_tokps_total"] = 99.94
    summary = build_summary(rows)
    ordered = ordered_rows(rows)
    assert ordered[0]["model_size_label"] == "0.8b"
    assert ordered[0]["batch_size"] == 1
    assert summary["cells"][-1]["decode_tokps_total"] == 99.94
    assert display_rate(99.94) == "99.9"
    assert display_rate(100.0) == "100"

    first_pair = rows[0]["model_pair"]
    gpu_rows = [
        {**rows[0], "model_pair": first_pair, "device": "NVIDIA GeForce RTX 5090"},
        {**rows[0], "model_pair": first_pair, "device": "NVIDIA GeForce RTX 4090"},
    ]
    assert [item["device"] for item in ordered_rows(gpu_rows)] == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
    ]
