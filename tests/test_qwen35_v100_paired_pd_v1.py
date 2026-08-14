from __future__ import annotations

from bench.validate_qwen35_best_optimized_hf_v1 import validate_matrix
from bench.validate_qwen35_v100_paired_pd_v1 import (
    PAIRS,
    QWEN_CONTRACT,
    QWEN_LANE,
    QWEN_MATRIX,
    QWEN_ROUTES,
    _validate_candidate_row,
    validate_correctness_manifest,
    validate_provenance,
)
from tests.test_qwen35_best_optimized_hf_v1 import complete_rows, use_raw_graph
from tests.test_qwen35_paired_pd_v1 import candidate_row


def v100_qwen_rows() -> list[dict]:
    rows = complete_rows()
    for item in rows:
        use_raw_graph(item)
        item.update(
            {
                "benchmark_matrix": QWEN_MATRIX,
                "device": "Tesla V100-PCIE-32GB",
                "torch_version": "2.5.1+cu124",
                "torch_cuda_version": "12.4",
                "triton_version": "3.4.0",
                "qwen_conv_backend_requested": "fla_triton",
                "qwen_fast_path_available": False,
                "qwen_causal_conv1d_importable": False,
                "qwen_conv_backend_effective": "fla_triton",
                "causal_conv1d_version": None,
            }
        )
    return rows


def v100_candidate(batch: int) -> dict:
    item = candidate_row(PAIRS[0], batch, 128, 128)
    layer_indices = list(range(1, 24))
    item.update(
        {
            "benchmark_matrix": "qwen35_v100_paired_pd_v1",
            "device": "Tesla V100-PCIE-32GB",
            "gpu_arch": "sm_70",
            "gpu_compute_capability": [7, 0],
            "torch_version": "2.5.1+cu124",
            "torch_cuda_version": "12.4",
            "triton_version": "3.4.0",
            "causal_conv1d_version": None,
            "rwkv_native_graph_rkv_policy": "manual",
            "rwkv_native_graph_state_dtype": (
                "torch.float16" if batch == 8 else "torch.float32"
            ),
            "rwkv_native_graph_triton_fp16_state": batch == 8,
            "rwkv_native_graph_sm70_wagv_lora_selected": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_effective": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_selected_layers": (
                layer_indices if batch == 1 else []
            ),
            "rwkv_native_graph_sm70_wagv_lora_effective_layers": (
                layer_indices if batch == 1 else []
            ),
            "rwkv_native_graph_sm70_wagv_lora_effective_layer_count": (
                23 if batch == 1 else 0
            ),
            "rwkv_native_graph_sm70_wagv_lora_full_eligible_layers_effective": (
                batch == 1
            ),
            "rwkv_native_graph_fused_wavg_lora_selected": batch == 8,
            "rwkv_native_graph_fused_wavg_lora_effective": batch == 8,
            "rwkv_native_graph_fused_wavg_lora_selected_layers": (
                layer_indices if batch == 8 else []
            ),
            "rwkv_native_graph_fused_wavg_lora_effective_layers": (
                layer_indices if batch == 8 else []
            ),
            "rwkv_native_graph_fused_wavg_lora_effective_layer_count": (
                23 if batch == 8 else 0
            ),
            "rwkv_native_graph_fused_wavg_lora_full_eligible_layers_effective": (
                batch == 8
            ),
            "rwkv_native_graph_ada_wagv_bmm_requested": False,
            "rwkv_native_graph_ada_wagv_bmm_selected": False,
            "rwkv_native_graph_ada_wagv_bmm_effective": False,
            "rwkv_native_graph_ada_wagv_bmm_selected_layers": [],
            "rwkv_native_graph_ada_wagv_bmm_effective_layers": [],
            "rwkv_native_graph_ada_wagv_bmm_effective_layer_count": 0,
            "rwkv_native_graph_ada_wagv_bmm_full_model_effective": False,
        }
    )
    return item


def test_v100_qwen_profile_accepts_fla_triton_raw_graph() -> None:
    summary = validate_matrix(
        v100_qwen_rows(),
        expected_device="Tesla V100-PCIE-32GB",
        expected_pairs=PAIRS,
        expected_routes_by_pair=QWEN_ROUTES,
        expected_matrix=QWEN_MATRIX,
        expected_lane=QWEN_LANE,
        expected_conv_backend="fla_triton",
        expected_causal_conv1d_importable=False,
        expected_fast_path_available=False,
        nullable_runtime_fields=("causal_conv1d_version",),
        qwen_contract=QWEN_CONTRACT,
    )
    assert summary["status"] == "pass", summary["errors"]
    assert summary["reference_rows"] == 48


def test_v100_candidate_routes_are_exact_for_b1_and_b8() -> None:
    for batch in (1, 8):
        errors: list[str] = []
        _validate_candidate_row(
            v100_candidate(batch),
            expected_device="Tesla V100-PCIE-32GB",
            errors=errors,
        )
        assert errors == []


def test_v100_candidate_rejects_route_telemetry_pollution() -> None:
    item = v100_candidate(8)
    item["rwkv_native_graph_sm70_wagv_lora_effective"] = True
    errors: list[str] = []
    _validate_candidate_row(
        item,
        expected_device="Tesla V100-PCIE-32GB",
        errors=errors,
    )
    assert any("sm70_wagv_lora_effective" in error for error in errors)


def test_v100_evidence_parsers_fail_closed_on_malformed_inputs(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    correctness = validate_correctness_manifest(malformed, [])
    assert correctness["status"] == "fail"
    provenance = validate_provenance(
        candidate_path=malformed,
        reference_paths=[],
        candidate_rows=[],
        route_manifest_path=malformed,
        qwen_route_manifest_paths=[],
        correctness_manifest_path=malformed,
        runtime_lock_path=malformed,
        candidate_model_hashes_path=malformed,
    )
    assert provenance["status"] == "fail"
