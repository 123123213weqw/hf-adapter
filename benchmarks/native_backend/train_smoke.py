#!/usr/bin/env python3
"""Verify that installing the inference backend preserves exact autograd."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from rwkv7_hf.kernel_bridge import last_backend_route, use_rwkv7_backend
from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def run_step(model, ids, labels, backend: str, parameter):
    model.zero_grad(set_to_none=True)
    with use_rwkv7_backend(backend):
        output = model(
            input_ids=ids,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        output.loss.backward()
    route = last_backend_route()
    gradient = parameter.grad.detach().clone()
    return output.loss.detach(), output.logits.detach(), gradient, route


def main() -> int:
    args = arguments()
    device = torch.device(args.device)
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    torch.manual_seed(args.seed)
    model = RWKV7ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    parameter = model.model.layers[0].attn.r_proj.weight
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ids = torch.randint(
        1,
        int(model.config.vocab_size),
        (2, 17),
        generator=generator,
        device=device,
    )
    labels = ids.clone()
    labels[0, :3] = -100

    reference = run_step(model, ids, labels, "reference", parameter)
    automatic = run_step(model, ids, labels, "auto", parameter)
    route = automatic[3]
    route_passed = bool(
        route
        and route.get("selected") == "reference"
        and "inference-only" in route.get("reason", "")
    )
    loss_equal = bool(torch.equal(automatic[0], reference[0]))
    logits_equal = bool(torch.equal(automatic[1], reference[1]))
    gradients_equal = bool(torch.equal(automatic[2], reference[2]))
    finite = bool(
        torch.isfinite(automatic[0])
        and torch.isfinite(automatic[1]).all()
        and torch.isfinite(automatic[2]).all()
    )
    nonzero_gradient = bool(automatic[2].abs().max() > 0)

    before = parameter.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    optimizer.step()
    parameter_changed = bool(not torch.equal(parameter.detach(), before))
    passed = bool(
        route_passed
        and loss_equal
        and logits_equal
        and gradients_equal
        and finite
        and nonzero_gradient
        and parameter_changed
    )
    report = {
        "schema": "rwkv7-native-backend-training-smoke-v1",
        "code_sha": git_sha(),
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
        },
        "gradient_checkpointing": bool(model.is_gradient_checkpointing),
        "route": route,
        "loss": float(automatic[0]),
        "gradient_max_abs": float(automatic[2].abs().max()),
        "checks": {
            "route_reference": route_passed,
            "loss_equal": loss_equal,
            "logits_equal": logits_equal,
            "gradients_equal": gradients_equal,
            "finite": finite,
            "nonzero_gradient": nonzero_gradient,
            "parameter_changed": parameter_changed,
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": passed}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
