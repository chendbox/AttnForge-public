#!/usr/bin/env python3
"""
Minimal kernel launcher for Nsight Compute profiling.

ncu will intercept CUDA kernel launches from this script.
We run one forward pass per backend — no warmup loops, no timing.

Usage (run via ncu, not directly):
    ncu --set full -o profile_v0_prefill \
        python benchmarks/profile_kernels.py --kernel prefill

    ncu --set full -o profile_sdpa_prefill \
        python benchmarks/profile_kernels.py --kernel prefill --backend sdpa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.backends import get_prefill_backend, get_decode_backend

# Llama-3.2-1B attention shape
BATCH      = 1
NUM_HEADS  = 32
SEQ_LEN    = 1024   # prefill
CTX_LEN    = 1024   # decode context
HEAD_DIM   = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM


def run_prefill(backend_name: str):
    if backend_name == "sdpa":
        q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        v = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        backend = get_prefill_backend(backend_name)
        q = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        k = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        v = torch.randn(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            backend.prefill(q, k, v, causal=True)


def run_decode(backend_name: str):
    if backend_name == "sdpa":
        q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.float32)
        k = torch.randn(BATCH, NUM_HEADS, CTX_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        v = torch.randn(BATCH, NUM_HEADS, CTX_LEN, HEAD_DIM, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            F.scaled_dot_product_attention(q, k, v, is_causal=False)
    else:
        backend = get_decode_backend(backend_name)
        q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.float32)
        k_cache = torch.randn(BATCH, CTX_LEN + 32, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        v_cache = torch.randn(BATCH, CTX_LEN + 32, HIDDEN_DIM, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            backend.decode(q, k_cache, v_cache, CTX_LEN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["prefill", "decode"], required=True)
    parser.add_argument("--backend", default="flashattn_v0",
                        help="flashattn_v0 | sdpa | sdpa_naive")
    args = parser.parse_args()

    torch.cuda.synchronize()

    if args.kernel == "prefill":
        run_prefill(args.backend)
    else:
        run_decode(args.backend)

    torch.cuda.synchronize()
    print(f"[profile_kernels] {args.kernel} / {args.backend} done.")


if __name__ == "__main__":
    main()
