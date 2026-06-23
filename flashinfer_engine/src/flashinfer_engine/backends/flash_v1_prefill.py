"""
FlashAttention v1 — prefill backend.

v1 = bf16 input + Tensor Cores (wmma):
  - bfloat16 Q/K/V input (halves memory bandwidth vs v0 fp32)
  - wmma 16×16×16 tiles for QK^T and PV matmuls (Tensor Core path)
  - fp32 softmax accumulators (precision)
  - MHA only (GQA support planned for v2)

Build: python -m csrc.compile --v1-only
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

_ENGINE_ROOT   = Path(__file__).parent.parent.parent.parent
_V1_KERNEL_DIR = _ENGINE_ROOT / "csrc" / "flash_v1_prefill"
if str(_V1_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_V1_KERNEL_DIR))

try:
    import custom_flash_attention_v1 as _ext
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class FlashV1PrefillBackend:
    """v1 prefill backend. bf16 input, wmma Tensor Cores, fp32 output."""

    def available(self) -> bool:
        return _AVAILABLE

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            q/k/v: (batch, num_heads, seq_len, head_dim)  — any float dtype
        Returns:
            out:   (batch, num_heads, seq_len, head_dim)  — fp32
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v1 prefill kernel not compiled.")

        batch, num_heads, seq_len, head_dim = q.shape
        hidden_dim = num_heads * head_dim

        # v1 kernel requires bf16 input
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        # (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        q_ = q.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        k_ = k.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        v_ = v.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)

        out_ = _ext.custom_flash_attention_v1(q_, k_, v_, num_heads, causal)

        # (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        return out_.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    def decode(self, *args, **kwargs):
        raise NotImplementedError("v1 prefill backend is prefill-only.")
