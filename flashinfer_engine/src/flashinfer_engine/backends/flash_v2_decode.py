"""
FlashAttention v2 decode backend.

v2 scope:
  - fp32 single-token decode
  - GQA-native KV cache layout: (batch, num_kv_heads, max_seq_len, head_dim)
  - Static cache only; paged KV cache remains a later runtime milestone
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


_ENGINE_ROOT = Path(__file__).parent.parent.parent.parent
_V2_KERNEL_DIR = _ENGINE_ROOT / "csrc" / "flash_v2_decode"
if str(_V2_KERNEL_DIR) not in sys.path:
    sys.path.insert(0, str(_V2_KERNEL_DIR))

try:
    import custom_flash_attention_v2_decode as _ext

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class FlashV2DecodeBackend:
    """GQA-native decode backend. fp32, static cache, single-token q."""

    def available(self) -> bool:
        return _AVAILABLE

    def prefill(self, *args, **kwargs):
        raise NotImplementedError("v2 decode backend is decode-only.")

    def update_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: int,
    ) -> None:
        """
        Append one decode-step K/V into a GQA-native cache.

        k_cache/v_cache: (batch, num_kv_heads, max_seq_len, head_dim)
        k/v:             (batch, num_kv_heads, 1, head_dim)
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v2 decode kernel not available (not compiled).")
        _ext.update_gqa_kv_cache(
            k_cache.contiguous(),
            v_cache.contiguous(),
            k.contiguous(),
            v.contiguous(),
            pos,
        )

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        """
        Args:
            q:          (batch, num_q_heads, 1, head_dim)
            k_cache:    (batch, num_kv_heads, max_seq_len, head_dim)
            v_cache:    (batch, num_kv_heads, max_seq_len, head_dim)
            context_len: number of valid tokens in the cache
        Returns:
            out:        (batch, num_q_heads, 1, head_dim)
        """
        if not _AVAILABLE:
            raise RuntimeError("flash_v2 decode kernel not available (not compiled).")
        return _ext.custom_flash_attention_v2_decode(
            q.contiguous(),
            k_cache.contiguous(),
            v_cache.contiguous(),
            context_len,
        )
