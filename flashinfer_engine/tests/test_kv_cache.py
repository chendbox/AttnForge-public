"""Decode backend + KV cache correctness tests.

These tests verify two things:
  1. The decode kernel produces the same output as SDPA when given the same K/V
     (functional correctness of single-token attention).
  2. Incremental decoding via a KV cache produces the same per-step output as
     re-running full attention from scratch each step (semantic correctness of
     the cache contract).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer_engine.backends import (
    available_decode_backends,
    get_decode_backend,
)
from flashinfer_engine.backends.flash_v0_decode import FlashV0DecodeBackend
from .conftest import TOL_FP32


# ---------- helpers ----------

def _sdpa_decode_reference(q, k, v) -> torch.Tensor:
    """Reference single-token decode: q is one position, k/v are full history.

    All shapes are (batch, heads, seq, head_dim).
    `q.shape[2] == 1`, `k.shape[2] == v.shape[2] == context_len`.
    Decode is non-causal because the single Q is the "current" token and
    every cached K/V is in the past.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


@pytest.fixture
def gqa_groups():
    """task5-style v0 decode kernel is MHA-only — match Q heads = KV heads."""
    return 1


# ---------- 1) Per-step output matches SDPA on the same K/V ----------

@pytest.mark.gpu
@pytest.mark.parametrize("backend_name", available_decode_backends() or ["sdpa_naive"])
@pytest.mark.parametrize("context_len", [16, 64, 256])
@pytest.mark.parametrize("num_heads,head_dim", [(8, 64), (4, 64)])
def test_decode_matches_sdpa(backend_name, context_len, num_heads, head_dim, device):
    """Decode output must match SDPA when both are given identical Q/K/V."""
    if backend_name not in available_decode_backends():
        pytest.skip(f"backend {backend_name} not available")

    backend = get_decode_backend(backend_name)
    batch = 1
    hidden_dim = num_heads * head_dim

    q = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=torch.float32)

    if isinstance(backend, FlashV0DecodeBackend):
        # v0 kernel uses flat (batch, max_seq_len, hidden) cache layout.
        max_len = context_len + 32
        k_cache = torch.randn(batch, max_len, hidden_dim, device=device, dtype=torch.float32)
        v_cache = torch.randn(batch, max_len, hidden_dim, device=device, dtype=torch.float32)
        # Build the SDPA reference from the SAME data sliced and reshaped:
        k_ref = (k_cache[:, :context_len, :]
                 .view(batch, context_len, num_heads, head_dim)
                 .transpose(1, 2)
                 .contiguous())
        v_ref = (v_cache[:, :context_len, :]
                 .view(batch, context_len, num_heads, head_dim)
                 .transpose(1, 2)
                 .contiguous())
    else:
        # SDPA naive uses (batch, heads, seq, head_dim) directly.
        k_cache = torch.randn(batch, num_heads, context_len, head_dim,
                              device=device, dtype=torch.float32)
        v_cache = torch.randn(batch, num_heads, context_len, head_dim,
                              device=device, dtype=torch.float32)
        k_ref, v_ref = k_cache, v_cache

    expected = _sdpa_decode_reference(q, k_ref, v_ref)
    actual = backend.decode(q, k_cache, v_cache, context_len)

    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, **TOL_FP32), (
        f"{backend_name} decode disagrees with SDPA reference at ctx={context_len} "
        f"(max abs diff = {(actual - expected).abs().max().item():.6f})"
    )


# ---------- 2) KV-cache semantic correctness ----------

@pytest.mark.gpu
def test_v0_update_kv_cache_writes_correct_position(device):
    """`update_kv_cache(pos)` must write k/v exactly at row `pos` of the cache."""
    if "flashattn_v0" not in available_decode_backends():
        pytest.skip("v0 decode kernel not available")

    backend = get_decode_backend("flashattn_v0")
    assert isinstance(backend, FlashV0DecodeBackend)

    batch, max_len, hidden = 2, 64, 256
    k_cache = torch.zeros(batch, max_len, hidden, device=device, dtype=torch.float32)
    v_cache = torch.zeros(batch, max_len, hidden, device=device, dtype=torch.float32)

    new_k = torch.randn(batch, 1, hidden, device=device, dtype=torch.float32)
    new_v = torch.randn(batch, 1, hidden, device=device, dtype=torch.float32)

    pos = 17
    backend.update_kv_cache(k_cache, v_cache, new_k, new_v, pos)

    # The target row matches.
    assert torch.allclose(k_cache[:, pos:pos + 1, :], new_k, **TOL_FP32)
    assert torch.allclose(v_cache[:, pos:pos + 1, :], new_v, **TOL_FP32)

    # All other rows are still zero (no spillover).
    other_rows_k = torch.cat([k_cache[:, :pos, :], k_cache[:, pos + 1:, :]], dim=1)
    other_rows_v = torch.cat([v_cache[:, :pos, :], v_cache[:, pos + 1:, :]], dim=1)
    assert (other_rows_k == 0).all(), "update_kv_cache wrote outside target position"
    assert (other_rows_v == 0).all(), "update_kv_cache wrote outside target position"


@pytest.mark.gpu
def test_v0_incremental_decode_matches_full_recompute(device):
    """The whole point of a KV cache.

    Incrementally append M new K/V into the cache one step at a time, comparing
    each step's decode output against a fresh SDPA run over the full prefix.
    If they diverge, the cache contract is broken.
    """
    if "flashattn_v0" not in available_decode_backends():
        pytest.skip("v0 decode kernel not available")

    backend = get_decode_backend("flashattn_v0")
    batch = 1
    num_heads, head_dim = 8, 64
    hidden = num_heads * head_dim
    prompt_len = 16     # initial prefix already in cache
    gen_len = 8         # number of decode steps to simulate
    max_len = 64

    # Pre-fill cache with the prompt.
    k_cache = torch.zeros(batch, max_len, hidden, device=device, dtype=torch.float32)
    v_cache = torch.zeros(batch, max_len, hidden, device=device, dtype=torch.float32)
    k_cache[:, :prompt_len, :] = torch.randn(
        batch, prompt_len, hidden, device=device, dtype=torch.float32,
    )
    v_cache[:, :prompt_len, :] = torch.randn(
        batch, prompt_len, hidden, device=device, dtype=torch.float32,
    )

    # Drive a sequence of decode steps.
    for step in range(gen_len):
        pos = prompt_len + step
        new_k = torch.randn(batch, 1, hidden, device=device, dtype=torch.float32)
        new_v = torch.randn(batch, 1, hidden, device=device, dtype=torch.float32)
        backend.update_kv_cache(k_cache, v_cache, new_k, new_v, pos)

        q = torch.randn(batch, num_heads, 1, head_dim, device=device, dtype=torch.float32)
        ctx = pos + 1

        # v0 backend output
        out_v0 = backend.decode(q, k_cache, v_cache, ctx)

        # Reference: SDPA on the (now-extended) cache reshape
        k_ref = (k_cache[:, :ctx, :]
                 .view(batch, ctx, num_heads, head_dim)
                 .transpose(1, 2).contiguous())
        v_ref = (v_cache[:, :ctx, :]
                 .view(batch, ctx, num_heads, head_dim)
                 .transpose(1, 2).contiguous())
        out_ref = _sdpa_decode_reference(q, k_ref, v_ref)

        assert torch.allclose(out_v0, out_ref, **TOL_FP32), (
            f"Step {step} (ctx={ctx}): cached decode != full recompute "
            f"(max abs diff = {(out_v0 - out_ref).abs().max().item():.6f})"
        )


# ---------- output sanity ----------

@pytest.mark.gpu
def test_decode_no_nans():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for backend_name in available_decode_backends():
        backend = get_decode_backend(backend_name)
        num_heads, head_dim, ctx = 8, 64, 128
        hidden = num_heads * head_dim
        q = torch.randn(1, num_heads, 1, head_dim, device="cuda")
        if isinstance(backend, FlashV0DecodeBackend):
            k = torch.randn(1, ctx + 8, hidden, device="cuda")
            v = torch.randn(1, ctx + 8, hidden, device="cuda")
        else:
            k = torch.randn(1, num_heads, ctx, head_dim, device="cuda")
            v = torch.randn(1, num_heads, ctx, head_dim, device="cuda")
        out = backend.decode(q, k, v, ctx)
        assert torch.isfinite(out).all(), f"{backend_name} produced non-finite output"
