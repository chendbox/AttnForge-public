"""
Backend registry.

Naming convention:
  - "sdpa"           — PyTorch baseline (cuDNN FlashAttention-2 internally)
  - "flashattn_v0"   — our custom kernel, baseline educational implementation
  - "flashattn_v1"   — (planned) bf16 + Tensor Cores
  - "flashattn_v2"   — (planned) GQA-native
  - "flashattn_v3"   — (planned) Paged KV Cache

Each new kernel version registers under both prefill and decode dicts.
"""
from __future__ import annotations

from .sdpa import SDPABackend
from .flash_v0_prefill import FlashV0PrefillBackend
from .flash_v0_decode import FlashV0DecodeBackend
from .flash_v1_prefill import FlashV1PrefillBackend

# Registry: name -> backend class
_PREFILL_BACKENDS: dict[str, type] = {
    "sdpa": SDPABackend,
    "flashattn_v0": FlashV0PrefillBackend,
    "flashattn_v1": FlashV1PrefillBackend,
}

_DECODE_BACKENDS: dict[str, type] = {
    "sdpa_naive": SDPABackend,
    "flashattn_v0": FlashV0DecodeBackend,
}


def get_prefill_backend(name: str):
    cls = _PREFILL_BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown prefill backend '{name}'. Available: {list(_PREFILL_BACKENDS)}")
    b = cls()
    if not b.available():
        raise RuntimeError(f"Backend '{name}' is not available (kernel not compiled?)")
    return b


def get_decode_backend(name: str):
    cls = _DECODE_BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown decode backend '{name}'. Available: {list(_DECODE_BACKENDS)}")
    b = cls()
    if not b.available():
        raise RuntimeError(f"Backend '{name}' is not available (kernel not compiled?)")
    return b


def available_prefill_backends() -> list[str]:
    return [name for name, cls in _PREFILL_BACKENDS.items() if cls().available()]


def available_decode_backends() -> list[str]:
    return [name for name, cls in _DECODE_BACKENDS.items() if cls().available()]
