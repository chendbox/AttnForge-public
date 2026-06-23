"""
FlashAttention v0 — prefill kernel.

v0 = baseline educational implementation:
  - fp32 only
  - MHA only (no GQA)
  - Tiled FlashAttention-1 style, no Tensor Cores
  - Causal masking supported

Future versions:
  v1 — bf16 + Tensor Cores (wmma)
  v2 — GQA-native (avoid KV repeat_interleave)
  v3 — Paged KV Cache integration
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

# The v0 prefill kernel .so lives in csrc/flash_v0_prefill/ (compiled in-place).
# Build with: python -m csrc.compile --prefill-only
_ENGINE_ROOT = Path(__file__).parent.parent.parent.parent
_V0_KERNEL_DIR = _ENGINE_ROOT / "csrc" / "flash_v0_prefill"
if str(_V0_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_V0_KERNEL_DIR))

try:
    import custom_flash_attention as _ext
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class FlashV0PrefillBackend:
    """v0 prefill backend wrapper. fp32, MHA-only."""

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
            q/k/v: (batch, num_heads, seq_len, head_dim)
        Returns:
            out:   (batch, num_heads, seq_len, head_dim)

        The v0 kernel expects (batch, seq, hidden_dim), so we transpose and merge heads.
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v0 prefill kernel not available (not compiled).")

        batch, num_heads, seq_len, head_dim = q.shape
        hidden_dim = num_heads * head_dim

        # (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        q_ = q.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        k_ = k.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        v_ = v.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)

        # Signature: (Q, K, V, num_heads, causal) -> tensor
        out_ = _ext.custom_flash_attention(q_, k_, v_, num_heads, causal)

        # (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        return out_.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

    def decode(self, *args, **kwargs):
        raise NotImplementedError("v0 prefill backend is prefill-only.")
