#!/usr/bin/env python3
"""Exercise a self-contained HF directory with the optional backend installed."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha", help="source revision when .git is unavailable")
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def bridge_for_model(model):
    package = model.__class__.__module__.rsplit(".", 1)[0]
    return importlib.import_module(f"{package}.kernel_bridge")


def main() -> int:
    args = arguments()
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    bridge = bridge_for_model(model)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    vocab_size = int(model.config.vocab_size)
    ids = torch.randint(
        1, vocab_size, (2, 17), generator=generator, device=device
    )
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, :3] = False
    mask[1, -4:] = False

    with torch.inference_mode():
        prefill = model(input_ids=ids, attention_mask=mask, use_cache=True)
        inference_route = bridge.last_backend_route()
        decode = model(
            input_ids=ids[:, -1:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        generated = model.generate(
            ids[:1],
            attention_mask=torch.ones_like(ids[:1]),
            max_new_tokens=8,
            num_beams=2,
            do_sample=False,
            use_cache=True,
        )

    save_dir = args.output.parent / "saved-model"
    model.save_pretrained(save_dir, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(
        save_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    reloaded.eval()
    with torch.inference_mode():
        reloaded_logits = reloaded(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
        ).logits
    reload_equal = bool(torch.equal(reloaded_logits, prefill.logits))
    del reloaded

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.zero_grad(set_to_none=True)
    training = model(
        input_ids=ids,
        attention_mask=mask,
        labels=ids,
        use_cache=False,
    )
    training.loss.backward()
    training_route = bridge.last_backend_route()
    gradients = [
        value.grad
        for value in model.parameters()
        if value.requires_grad and value.grad is not None
    ]
    finite_gradients = bool(
        gradients and all(torch.isfinite(value).all() for value in gradients)
    )
    nonzero_gradient = bool(
        gradients and any(value.detach().abs().max() > 0 for value in gradients)
    )

    inference_routed = bool(
        inference_route and inference_route.get("selected") == "optimized"
    )
    training_fell_back = bool(
        training_route
        and training_route.get("selected") == "reference"
        and "inference-only" in training_route.get("reason", "")
    )
    finite = bool(
        torch.isfinite(prefill.logits).all()
        and torch.isfinite(decode.logits).all()
        and torch.isfinite(training.loss)
    )
    passed = bool(
        inference_routed
        and training_fell_back
        and finite
        and finite_gradients
        and nonzero_gradient
        and reload_equal
        and generated.shape == (1, ids.shape[1] + 8)
    )
    report = {
        "schema": "rwkv7-native-backend-hf-ecosystem-smoke-v1",
        "code_sha": args.code_sha or git_sha(),
        "model": str(args.model.resolve()),
        "dtype": args.dtype,
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
            "dynamic_module": model.__class__.__module__,
        },
        "inference_route": inference_route,
        "training_route": training_route,
        "checks": {
            "auto_model_loaded": True,
            "optimized_inference": inference_routed,
            "cached_decode": tuple(decode.logits.shape[:2]) == (2, 1),
            "beam_generation": generated.shape == (1, ids.shape[1] + 8),
            "save_reload_equal": reload_equal,
            "training_reference_fallback": training_fell_back,
            "finite": finite,
            "finite_gradients": finite_gradients,
            "nonzero_gradient": nonzero_gradient,
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
