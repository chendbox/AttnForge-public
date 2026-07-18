"""Focused tests for the v1 P @ V WMMA debug tile."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from .conftest import TOL_BF16

_ENGINE_ROOT = Path(__file__).parent.parent
_V1_KERNEL_DIR = _ENGINE_ROOT / "csrc" / "flash_v1_prefill"
if str(_V1_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_KERNEL_DIR))

try:
    import custom_flash_attention_v1 as _ext
except ImportError:
    _ext = None


@pytest.mark.gpu
def test_debug_pv_wmma_matches_torch(device):
    if _ext is None:
        pytest.skip("flash_v1 extension is not compiled")

    p = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
    v = torch.randn(16, 16, device=device, dtype=torch.bfloat16)

    actual = _ext.debug_pv_wmma(p, v)
    expected = p.float() @ v.float()

    assert torch.allclose(actual, expected, **TOL_BF16), (
        f"debug_pv_wmma max diff = {(actual - expected).abs().max().item():.6f}"
    )
