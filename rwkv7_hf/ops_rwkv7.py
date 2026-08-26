# coding=utf-8
"""Readable RWKV-7 recurrence plus one optional optimized boundary."""
from __future__ import annotations

import torch

try:
    from .kernel_bridge import try_optimized_recurrent
except ImportError:  # package-free Hugging Face remote-code execution
    from kernel_bridge import try_optimized_recurrent


def _validate_recurrent_inputs(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[int, int, int, int, int, torch.Tensor | None]:
    if receptance.ndim != 4:
        raise ValueError("RWKV7 recurrent inputs must be shaped [B,T,H,D]")
    if any(tensor.ndim != 4 for tensor in (decay, key, value, a, b)):
        raise ValueError("RWKV7 recurrent inputs must be shaped [B,T,H,D]")
    batch, time, heads, key_dim = receptance.shape
    value_dim = int(value.shape[-1])
    expected_key_shape = (batch, time, heads, key_dim)
    if any(tuple(tensor.shape) != expected_key_shape for tensor in (decay, key, a, b)):
        raise ValueError("receptance, decay, key, a, and b must have identical shapes")
    if tuple(value.shape[:3]) != (batch, time, heads):
        raise ValueError("value must share the [B,T,H] dimensions")
    if tuple(initial_state.shape) != (batch, heads, key_dim, value_dim):
        raise ValueError(
            "initial_state must be shaped [batch, heads, key_dim, value_dim]"
        )
    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch, time):
            raise ValueError("attention_mask must be shaped [batch, time]")
        attention_mask = attention_mask.to(
            device=initial_state.device, dtype=torch.bool
        )
    return batch, time, heads, key_dim, value_dim, attention_mask


def rwkv7_recurrent_reference(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the readable PyTorch recurrence in canonical ``[K,V]`` layout."""

    batch, time, _, _, _, attention_mask = _validate_recurrent_inputs(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )

    state = initial_state
    outputs: list[torch.Tensor] = []
    for token_idx in range(time):
        # Match the official reference's mixed-precision contract exactly:
        # projections and their outer products stay in the model dtype, while
        # the accumulated recurrent state and decay are FP32.  Casting every
        # operand to the state dtype here looks numerically attractive, but it
        # is a different model in FP16/BF16 and makes a converted checkpoint
        # diverge from the official token-wise implementation.
        r_t = receptance[:, token_idx]
        w_t = decay[:, token_idx].to(dtype=state.dtype)
        k_t = key[:, token_idx]
        v_t = value[:, token_idx]
        a_t = a[:, token_idx]
        b_t = b[:, token_idx]

        # The public/cache layout is [K,V]. Evaluate the equation in the
        # official [V,K] presentation, then transpose the result back. Keeping
        # this multiplication order is important for numerical parity with the
        # official RWKV implementation over long recurrent sequences.
        state_vk = state.transpose(-1, -2)
        ab = a_t.unsqueeze(-1) @ b_t.unsqueeze(-2)
        vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
        candidate_vk = (
            state_vk * w_t.unsqueeze(-2)
            + state_vk @ ab.to(dtype=state.dtype)
            + vk.to(dtype=state.dtype)
        )
        candidate = candidate_vk.transpose(-1, -2)
        output = (
            candidate_vk.to(dtype=r_t.dtype) @ r_t.unsqueeze(-1)
        ).squeeze(-1)

        if attention_mask is not None:
            active = attention_mask[:, token_idx]
            state_active = active.view(batch, 1, 1, 1)
            output_active = active.view(batch, 1, 1)
            state = torch.where(state_active, candidate, state)
            output = torch.where(output_active, output, torch.zeros_like(output))
        else:
            state = candidate
        outputs.append(output.to(dtype=value.dtype))

    return torch.stack(outputs, dim=1), state


def _validate_optimized_result(
    result,
    *,
    receptance: torch.Tensor,
    value: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("optimized RWKV7 recurrence must return (output, final_state)")
    output, final_state = result
    if not isinstance(output, torch.Tensor) or not isinstance(final_state, torch.Tensor):
        raise TypeError("optimized RWKV7 recurrence must return two tensors")
    expected_output = (*receptance.shape[:3], value.shape[-1])
    if tuple(output.shape) != tuple(expected_output):
        raise RuntimeError(
            f"optimized RWKV7 output shape {tuple(output.shape)} != {expected_output}"
        )
    if tuple(final_state.shape) != tuple(initial_state.shape):
        raise RuntimeError(
            "optimized RWKV7 final-state shape "
            f"{tuple(final_state.shape)} != {tuple(initial_state.shape)}"
        )
    if output.device != value.device or final_state.device != initial_state.device:
        raise RuntimeError("optimized RWKV7 backend returned tensors on the wrong device")
    if output.dtype != value.dtype or final_state.dtype != initial_state.dtype:
        raise RuntimeError("optimized RWKV7 backend returned tensors with the wrong dtype")
    return output, final_state


def rwkv7_recurrent(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run RWKV-7 with an optional Mamba-style optimized operator.

    The surrounding model and cache semantics do not change. ``auto`` uses an
    installed versioned companion implementation only when that package
    explicitly accepts the complete request; otherwise the readable
    PyTorch implementation remains the fallback.
    """

    _, _, _, _, _, attention_mask = _validate_recurrent_inputs(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )
    optimized = try_optimized_recurrent(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
        backend=backend,
        training=training,
    )
    if optimized is not None:
        return _validate_optimized_result(
            optimized,
            receptance=receptance,
            value=value,
            initial_state=initial_state,
        )
    return rwkv7_recurrent_reference(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )


__all__ = ["rwkv7_recurrent", "rwkv7_recurrent_reference"]
