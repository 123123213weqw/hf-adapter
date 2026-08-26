from __future__ import annotations

import torch

from kernel_wheel.rwkv7_kernels.recurrent_graph import _reference_recurrent


def test_graph_math_is_invariant_to_batch_regrouping() -> None:
    torch.manual_seed(41)
    batch, time, heads, width = 8, 3, 1, 8
    tensors = [
        torch.randn(batch, time, heads, width, dtype=torch.float16)
        for _ in range(6)
    ]
    state = torch.randn(batch, heads, width, width, dtype=torch.float32)
    mask = torch.ones(batch, time, dtype=torch.bool)
    mask[5, 0] = False

    grouped = _reference_recurrent(*tensors, state, mask)
    isolated = _reference_recurrent(
        *(value[5:6] for value in tensors),
        state[5:6],
        mask[5:6],
    )

    torch.testing.assert_close(grouped[0][5:6], isolated[0], rtol=0, atol=0)
    torch.testing.assert_close(grouped[1][5:6], isolated[1], rtol=0, atol=0)
