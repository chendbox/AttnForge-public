"""Correctness tests for the v2 GQA-native decode backend."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer_engine.backends import available_decode_backends, get_decode_backend


TOL_FP32 = {"rtol": 1e-3, "atol": 1e-2}


def _sdpa_gqa_reference(q, k_cache, v_cache, context_len):
    """Expand GQA K/V heads only for the PyTorch SDPA reference."""
    num_q_heads = q.shape[1]
    k = k_cache[:, :, :context_len, :]
    v = v_cache[:, :, :context_len, :]
    if k.shape[1] != num_q_heads:
        groups = num_q_heads // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


@pytest.mark.gpu
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("context_len", [16, 64, 257])
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [(8, 2, 64), (32, 8, 64)])
def test_v2_gqa_decode_matches_sdpa(batch, context_len, num_q_heads, num_kv_heads, head_dim, device):
    if "flashattn_v2" not in available_decode_backends():
        pytest.skip("flashattn_v2 decode kernel not available")

    backend = get_decode_backend("flashattn_v2")
    max_len = context_len + 8
    q = torch.randn(batch, num_q_heads, 1, head_dim, device=device, dtype=torch.float32)
    k_cache = torch.randn(batch, num_kv_heads, max_len, head_dim, device=device, dtype=torch.float32)
    v_cache = torch.randn(batch, num_kv_heads, max_len, head_dim, device=device, dtype=torch.float32)

    expected = _sdpa_gqa_reference(q, k_cache, v_cache, context_len)
    actual = backend.decode(q, k_cache, v_cache, context_len)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, **TOL_FP32), (
        f"flashattn_v2 decode disagrees with SDPA "
        f"(max abs diff = {(actual - expected).abs().max().item():.6f})"
    )


@pytest.mark.gpu
def test_v2_update_gqa_kv_cache_writes_correct_position(device):
    if "flashattn_v2" not in available_decode_backends():
        pytest.skip("flashattn_v2 decode kernel not available")

    backend = get_decode_backend("flashattn_v2")
    batch, num_kv_heads, max_len, head_dim = 2, 4, 64, 32
    k_cache = torch.zeros(batch, num_kv_heads, max_len, head_dim, device=device, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    new_k = torch.randn(batch, num_kv_heads, 1, head_dim, device=device, dtype=torch.float32)
    new_v = torch.randn(batch, num_kv_heads, 1, head_dim, device=device, dtype=torch.float32)

    pos = 17
    backend.update_kv_cache(k_cache, v_cache, new_k, new_v, pos)

    assert torch.allclose(k_cache[:, :, pos:pos + 1, :], new_k, **TOL_FP32)
    assert torch.allclose(v_cache[:, :, pos:pos + 1, :], new_v, **TOL_FP32)
    assert (k_cache[:, :, :pos, :] == 0).all()
    assert (k_cache[:, :, pos + 1:, :] == 0).all()
    assert (v_cache[:, :, :pos, :] == 0).all()
    assert (v_cache[:, :, pos + 1:, :] == 0).all()
