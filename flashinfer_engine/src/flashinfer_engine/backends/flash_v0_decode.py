"""
FlashAttention v0 — decode kernel + KV cache update.

v0 = baseline educational implementation:
  - fp32 only
  - MHA only (no GQA)
  - Static KV cache (no paging)
  - Single-token decode against full cached context

Future versions:
  v1 — bf16 + Tensor Cores (wmma)
  v2 — GQA-native KV cache
  v3 — Paged KV Cache (block manager)
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch

# The v0 decode kernel .so lives in csrc/flash_v0_decode/ (compiled in-place).
# Build with: python -m csrc.compile --decode-only
_ENGINE_ROOT = Path(__file__).parent.parent.parent.parent
_V0_KERNEL_DIR = _ENGINE_ROOT / "csrc" / "flash_v0_decode"
if str(_V0_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_V0_KERNEL_DIR))

try:
    import custom_flash_attention_decode as _ext
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class FlashV0DecodeBackend:
    """v0 decode backend wrapper. fp32, MHA-only, static cache."""

    def available(self) -> bool:
        return _AVAILABLE

    def prefill(self, *args, **kwargs):
        raise NotImplementedError("v0 decode backend is decode-only.")

    def update_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
    ) -> None:
        """
        Append a single decode-step K/V into the cache at position `pos`.

        k_cache/v_cache: (batch, max_seq_len, hidden_dim)
        k/v:             (batch, 1, hidden_dim)
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v0 decode kernel not available (not compiled).")
        _ext.update_kv_cache(k_cache, v_cache, k.contiguous(), v.contiguous(), pos)

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        """
        Args:
            q:          (batch, num_heads, 1, head_dim)  — single new token
            k_cache:    (batch, max_seq_len, hidden_dim) — flat cache
            v_cache:    (batch, max_seq_len, hidden_dim)
            context_len: number of valid tokens in cache (including current token)
        Returns:
            out:        (batch, num_heads, 1, head_dim)
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v0 decode kernel not available (not compiled).")

        batch, num_heads, _, head_dim = q.shape
        hidden_dim = num_heads * head_dim

        # (batch, heads, 1, head_dim) -> (batch, 1, hidden)
        q_ = q.transpose(1, 2).contiguous().view(batch, 1, hidden_dim)

        # Slice cache to valid context (kernel expects only the populated portion).
        k_full = k_cache[:, :context_len, :].contiguous()
        v_full = v_cache[:, :context_len, :].contiguous()

        # Signature: (Q, K_full, V_full, num_heads, causal) -> tensor
        # causal=False: Q is a single new token, all cached K/V are past tokens.
        out_ = _ext.custom_flash_attention_decode(q_, k_full, v_full, num_heads, False)

        # (batch, 1, hidden) -> (batch, heads, 1, head_dim)
        return out_.view(batch, 1, num_heads, head_dim).transpose(1, 2)
