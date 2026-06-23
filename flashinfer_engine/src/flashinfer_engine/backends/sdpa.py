"""PyTorch scaled_dot_product_attention backend — always available, used as baseline."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class SDPABackend:
    def available(self) -> bool:
        return True

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
        """
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        """
        Naive decode: attend over full cached context with SDPA (no kernel optimisation).

        Args:
            q:          (batch, num_heads, 1, head_dim)
            k_cache:    (batch, num_heads, max_seq_len, head_dim)
            v_cache:    (batch, num_heads, max_seq_len, head_dim)
            context_len: number of valid tokens in cache
        Returns:
            out:        (batch, num_heads, 1, head_dim)
        """
        k = k_cache[:, :, :context_len, :]
        v = v_cache[:, :, :context_len, :]
        # causal=False: q is a single new token, all cache tokens are in the past
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)
