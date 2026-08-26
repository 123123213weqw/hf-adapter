#!/usr/bin/env python3
"""Validate an installed optional backend against the readable HF model."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from rwkv7_hf.kernel_bridge import (
    kernel_bridge_status,
    last_backend_route,
    reset_kernel_discovery_for_tests,
    use_rwkv7_backend,
)
from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent, rwkv7_recurrent_reference


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model", action="append", required=True, help="label=path")
    result.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batches", default="1,4")
    result.add_argument("--lengths", default="1,17,128")
    result.add_argument("--decode-tokens", type=int, default=64)
    result.add_argument("--seed", type=int, default=42)
    return result


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("--model must be label=path")
        label, path = value.split("=", 1)
        parsed.append((label, Path(path).expanduser().resolve()))
    return parsed


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual32 = actual.detach().float().reshape(-1)
    expected32 = expected.detach().float().reshape(-1)
    finite = bool(torch.isfinite(actual32).all() and torch.isfinite(expected32).all())
    if actual32.numel() == 0:
        cosine = 1.0
        max_abs = 0.0
    else:
        cosine = float(F.cosine_similarity(actual32, expected32, dim=0))
        max_abs = float((actual32 - expected32).abs().max())
    return {"finite": finite, "cosine": cosine, "max_abs": max_abs}


def passed_metrics(
    actual: torch.Tensor, expected: torch.Tensor, dtype_name: str, *, logits: bool
) -> bool:
    row = metrics(actual, expected)
    if not row["finite"]:
        return False
    if dtype_name == "fp32":
        return bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
    if row["cosine"] < 0.9999:
        return False
    return not (dtype_name == "fp16" and logits and row["max_abs"] > 0.15)


def expected_route(dtype_name: str) -> str:
    """Return the route promoted by the v1 companion package."""

    return "optimized" if dtype_name == "fp16" else "reference"


def route_passed(route: dict[str, str] | None, dtype_name: str) -> bool:
    return bool(route and route.get("selected") == expected_route(dtype_name))


def operator_matrix(
    *, dtype: torch.dtype, dtype_name: str, device: torch.device, seed: int
) -> list[dict[str, Any]]:
    rows = []
    for batch in (1, 4):
        for length in (1, 17, 128):
            generator = torch.Generator(device=device).manual_seed(
                seed + batch * 1000 + length
            )
            shape = (batch, length, 2, 64)
            values = [
                torch.randn(shape, generator=generator, device=device, dtype=dtype)
                * 0.05
                for _ in range(6)
            ]
            values[1] = torch.sigmoid(values[1].float()).to(dtype)
            values[4] = values[4] * 0.1
            values[5] = values[5] * 0.1
            state = (
                torch.randn(
                    batch,
                    2,
                    64,
                    64,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                * 0.01
            )
            mask = torch.ones(batch, length, device=device, dtype=torch.bool)
            if batch > 1 and length > 1:
                mask[1, : min(3, length)] = False
                mask[2, -min(4, length) :] = False
            with torch.inference_mode():
                reference = rwkv7_recurrent_reference(*values, state, mask)
                reset_kernel_discovery_for_tests()
                automatic = rwkv7_recurrent(*values, state, mask, backend="auto")
            route = last_backend_route()
            output_metrics = metrics(automatic[0], reference[0])
            state_metrics = metrics(automatic[1], reference[1])
            passed = passed_metrics(
                automatic[0], reference[0], dtype_name, logits=False
            ) and passed_metrics(
                automatic[1], reference[1], dtype_name, logits=False
            ) and route_passed(route, dtype_name)
            rows.append(
                {
                    "batch": batch,
                    "length": length,
                    "output": output_metrics,
                    "state": state_metrics,
                    "route": route,
                    "passed": bool(passed),
                }
            )
    return rows


def model_case(
    model,
    *,
    dtype_name: str,
    ids: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    with torch.inference_mode(), use_rwkv7_backend("reference"):
        reference = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
        )
    reset_kernel_discovery_for_tests()
    with torch.inference_mode(), use_rwkv7_backend("auto"):
        automatic = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
        )
    route = last_backend_route()
    recurrent_pairs = list(
        zip(
            automatic.past_key_values.recurrent_state,
            reference.past_key_values.recurrent_state,
        )
    )
    logits_passed = passed_metrics(
        automatic.logits, reference.logits, dtype_name, logits=True
    )
    states_passed = all(
        passed_metrics(actual, expected, dtype_name, logits=False)
        for actual, expected in recurrent_pairs
    )
    state_rows = [metrics(actual, expected) for actual, expected in recurrent_pairs]
    return {
        "batch": int(ids.shape[0]),
        "length": int(ids.shape[1]),
        "logits": metrics(automatic.logits, reference.logits),
        "state_min_cosine": min(row["cosine"] for row in state_rows),
        "state_max_abs": max(row["max_abs"] for row in state_rows),
        "route": route,
        "passed": bool(
            logits_passed and states_passed and route_passed(route, dtype_name)
        ),
    }


def cached_and_greedy(
    model,
    *,
    dtype_name: str,
    device: torch.device,
    vocab_size: int,
    decode_tokens: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed + 90_000)
    prompt = torch.randint(1, vocab_size, (1, 17), generator=generator, device=device)
    continuation = torch.randint(
        1,
        vocab_size,
        (1, decode_tokens),
        generator=generator,
        device=device,
    )
    mask = torch.ones_like(prompt, dtype=torch.bool)

    def teacher(backend: str):
        with torch.inference_mode(), use_rwkv7_backend(backend):
            output = model(
                input_ids=prompt,
                attention_mask=mask,
                use_cache=True,
            )
            cache = output.past_key_values
            pieces = []
            for index in range(decode_tokens):
                output = model(
                    input_ids=continuation[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                pieces.append(output.logits)
            logits = torch.cat(pieces, dim=1)
            generated = model.generate(
                prompt,
                attention_mask=mask,
                max_new_tokens=64,
                do_sample=False,
                use_cache=True,
            )
        return logits, cache, generated

    reference = teacher("reference")
    reset_kernel_discovery_for_tests()
    automatic = teacher("auto")
    state_pairs = list(
        zip(automatic[1].recurrent_state, reference[1].recurrent_state)
    )
    logits_passed = passed_metrics(
        automatic[0], reference[0], dtype_name, logits=True
    )
    states_passed = all(
        passed_metrics(actual, expected, dtype_name, logits=False)
        for actual, expected in state_pairs
    )
    greedy_equal = bool(torch.equal(automatic[2], reference[2]))
    route = last_backend_route()
    return {
        "teacher_forced_logits": metrics(automatic[0], reference[0]),
        "final_state_min_cosine": min(
            metrics(actual, expected)["cosine"] for actual, expected in state_pairs
        ),
        "greedy_equal": greedy_equal,
        "greedy_tokens": int(automatic[2].shape[1] - prompt.shape[1]),
        "route": route,
        "passed": bool(
            logits_passed
            and states_passed
            and greedy_equal
            and route_passed(route, dtype_name)
        ),
    }


def main() -> int:
    args = parser().parse_args()
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    batches = [int(value) for value in args.batches.split(",")]
    lengths = [int(value) for value in args.lengths.split(",")]
    models = parse_models(args.model)
    args.output.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = {
        "schema": "rwkv7-native-backend-validation-v1",
        "code_sha": git_sha(),
        "dtype": args.dtype,
        "expected_route": expected_route(args.dtype),
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "kernel_bridge": kernel_bridge_status(),
        },
        "operator": operator_matrix(
            dtype=dtype, dtype_name=args.dtype, device=device, seed=args.seed
        ),
        "models": {},
    }

    for model_index, (label, path) in enumerate(models):
        model = RWKV7ForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        vocab_size = int(model.config.vocab_size)
        rows = []
        for batch in batches:
            for length in lengths:
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + model_index * 100_000 + batch * 1000 + length
                )
                ids = torch.randint(
                    1, vocab_size, (batch, length), generator=generator, device=device
                )
                mask = torch.ones(batch, length, device=device, dtype=torch.bool)
                if batch > 1 and length > 1:
                    mask[1, : min(3, length)] = False
                    mask[2, -min(4, length) :] = False
                rows.append(
                    model_case(
                        model,
                        dtype_name=args.dtype,
                        ids=ids,
                        mask=mask,
                    )
                )
        generation = cached_and_greedy(
            model,
            dtype_name=args.dtype,
            device=device,
            vocab_size=vocab_size,
            decode_tokens=args.decode_tokens,
            seed=args.seed + model_index,
        )
        bundle["models"][label] = {
            "path": str(path),
            "config_sha256": sha256(path / "config.json"),
            "cases": rows,
            "cached_and_greedy": generation,
            "passed": bool(all(row["passed"] for row in rows) and generation["passed"]),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    operator_passed = all(row["passed"] for row in bundle["operator"])
    models_passed = all(row["passed"] for row in bundle["models"].values())
    bundle["passed"] = bool(operator_passed and models_passed)
    output = args.output / f"validation-{args.dtype}.json"
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "passed": bundle["passed"]}))
    return 0 if bundle["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
