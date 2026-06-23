"""Prefill backend correctness tests.

Every prefill backend must produce numerically equivalent output to PyTorch
SDPA (within fp32 tolerance) on random inputs.  This is the gate that
catches algorithmic regressions before a benchmark-only PR can hide them
behind plausible-looking latency numbers.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer_engine.backends import available_prefill_backends, get_prefill_backend
from .conftest import TOL_BF16, TOL_FP32


# ---------- reference implementation ----------

def _sdpa_reference(q, k, v, causal: bool) -> torch.Tensor:
    """Ground truth: PyTorch SDPA in fp32 (no Tensor Core, so deterministic)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _prefill_tolerance(backend_name: str) -> dict[str, float]:
    """v1 uses bf16 inputs internally, so it needs a looser comparison window."""
    if backend_name == "flashattn_v1":
        return TOL_BF16
    return TOL_FP32


def _expand_kv_if_gqa(k, v, num_q_heads):
    """task4-style kernel is MHA-only; expand K/V to match Q heads."""
    if k.shape[1] != num_q_heads:
        groups = num_q_heads // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return k, v


# ---------- parameter matrix ----------

# (batch, num_q_heads, seq_len, head_dim)
PREFILL_SHAPES = [
    (1, 4, 64, 64),     # tiny: smallest non-trivial shape
    (1, 8, 128, 64),    # single batch, modest seq
    (2, 8, 256, 64),    # multi-batch
    (1, 16, 512, 64),   # closer-to-real-world
]


@pytest.mark.gpu
@pytest.mark.parametrize("backend_name", available_prefill_backends() or ["sdpa"])
@pytest.mark.parametrize("shape", PREFILL_SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_prefill_matches_sdpa(backend_name, shape, causal, device):
    """Each prefill backend must agree with SDPA reference within fp32 tolerance."""
    if backend_name not in available_prefill_backends():
        pytest.skip(f"backend {backend_name} not available")

    batch, num_heads, seq_len, head_dim = shape

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    backend = get_prefill_backend(backend_name)

    expected = _sdpa_reference(q, k, v, causal=causal)
    actual = backend.prefill(q, k, v, causal=causal)
    tol = _prefill_tolerance(backend_name)

    assert actual.shape == expected.shape, (
        f"shape mismatch: backend={actual.shape}, sdpa={expected.shape}"
    )
    assert torch.allclose(actual, expected, **tol), (
        f"{backend_name} disagrees with SDPA "
        f"(max abs diff = {(actual - expected).abs().max().item():.6f})"
    )


@pytest.mark.gpu
def test_prefill_no_nans():
    """Sanity: backend output should never contain NaN/Inf for normal inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    for backend_name in available_prefill_backends():
        backend = get_prefill_backend(backend_name)
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn(1, 8, 128, 64, device="cuda")
        v = torch.randn(1, 8, 128, 64, device="cuda")
        out = backend.prefill(q, k, v, causal=True)
        assert torch.isfinite(out).all(), f"{backend_name} produced non-finite output"


@pytest.mark.gpu
def test_prefill_causal_mask_actually_masks():
    """A query at position i must not depend on K/V at positions j > i (causal)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    for backend_name in available_prefill_backends():
        backend = get_prefill_backend(backend_name)

        seq_len = 64
        q = torch.randn(1, 4, seq_len, 64, device="cuda")
        k = torch.randn(1, 4, seq_len, 64, device="cuda")
        v = torch.randn(1, 4, seq_len, 64, device="cuda")

        # Output for the first token must be independent of K/V positions > 0
        out1 = backend.prefill(q.clone(), k.clone(), v.clone(), causal=True)[:, :, 0, :]

        # Perturb K/V at later positions; first-token output must NOT change
        k2 = k.clone()
        v2 = v.clone()
        k2[:, :, 1:, :].add_(10.0)
        v2[:, :, 1:, :].add_(10.0)
        out2 = backend.prefill(q, k2, v2, causal=True)[:, :, 0, :]

        assert torch.allclose(out1, out2, atol=1e-4), (
            f"{backend_name}: first-token output changed after perturbing future K/V — "
            "causal mask is leaking"
        )
