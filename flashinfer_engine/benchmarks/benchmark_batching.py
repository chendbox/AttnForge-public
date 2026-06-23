#!/usr/bin/env python3
"""
Dynamic batching throughput benchmark.

Measures sequential (batch=1 × N) vs batched (batch=N × 1) attention
throughput across batch sizes and model shapes.

Uses *synthetic* Q/K/V tensors — no real model download required.
Works with any model yaml (llama3_2_1b, mistral_7b, etc.).

Usage:
    python benchmarks/benchmark_batching.py --model configs/models/llama3_2_1b.yaml
    python benchmarks/benchmark_batching.py --model configs/models/mistral_7b.yaml --backends sdpa
    python benchmarks/benchmark_batching.py \\
        --model configs/models/llama3_2_1b.yaml \\
        --backends sdpa flashattn_v0 \\
        --batch-sizes 1 2 4 8 16 32 \\
        --seq-len 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.backends import available_prefill_backends, get_prefill_backend
from flashinfer_engine.config import ModelConfig
from flashinfer_engine.metrics import measure_latency, percentile_stats

# ------------------------------------------------------------------ #
# Core benchmark logic
# ------------------------------------------------------------------ #

def _alloc_qkv(batch, num_heads, head_dim, seq_len, dtype, device):
    q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def run_batching_bench(
    backend_name: str,
    model: ModelConfig,
    batch_sizes: list[int],
    seq_len: int,
    dtype: torch.dtype,
    num_warmup: int,
    num_runs: int,
    device: torch.device,
) -> None:
    backend = get_prefill_backend(backend_name)
    num_heads = model.num_heads
    head_dim  = model.head_dim

    print(f"\n  Backend: {backend_name}")
    print(f"  {'Batch':>6}  {'Mode':>12}  {'p50 (ms)':>10}  {'tok/s':>10}  {'Speedup':>8}")
    print(f"  {'-'*55}")

    # ---- sequential baseline (batch=1, run N times) ------------------
    q1, k1, v1 = _alloc_qkv(1, num_heads, head_dim, seq_len, dtype, device)

    def seq_fn():
        with torch.no_grad():
            backend.prefill(q1, k1, v1, causal=True)

    seq_times = measure_latency(seq_fn, num_warmup=num_warmup, num_runs=num_runs)
    seq_stats  = percentile_stats(seq_times)
    seq_p50    = seq_stats["p50_ms"]
    seq_tps    = (1 * seq_len) / (seq_p50 / 1000.0)

    print(f"  {'1':>6}  {'sequential':>12}  {seq_p50:>10.3f}  {seq_tps:>10.0f}  {'1.00x':>8}")

    # ---- batched runs ------------------------------------------------
    for batch in batch_sizes:
        if batch == 1:
            continue  # already printed above

        qb, kb, vb = _alloc_qkv(batch, num_heads, head_dim, seq_len, dtype, device)

        def bat_fn():
            with torch.no_grad():
                backend.prefill(qb, kb, vb, causal=True)

        bat_times = measure_latency(bat_fn, num_warmup=num_warmup, num_runs=num_runs)
        bat_stats  = percentile_stats(bat_times)
        bat_p50    = bat_stats["p50_ms"]

        # throughput = total tokens processed / time
        bat_tps  = (batch * seq_len) / (bat_p50 / 1000.0)
        speedup  = bat_tps / seq_tps

        print(f"  {batch:>6}  {'batched':>12}  {bat_p50:>10.3f}  {bat_tps:>10.0f}  {speedup:>7.2f}x")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    parser = argparse.ArgumentParser(
        description="Sequential vs batched attention throughput benchmark"
    )
    parser.add_argument("--model", required=True,
                        help="Path to model yaml (e.g. configs/models/llama3_2_1b.yaml)")
    parser.add_argument("--backends", nargs="+", default=None,
                        help="Backends to test (default: all available prefill backends)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32],
                        help="Batch sizes to sweep (default: 1 2 4 8 16 32)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length (default: 512)")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-runs",   type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA GPU required.")

    device = torch.device("cuda")
    model  = ModelConfig.from_yaml(args.model)
    dtype  = _DTYPE_MAP[args.dtype]

    backends = args.backends or available_prefill_backends()
    if not backends:
        sys.exit("No prefill backends available. Use --backends sdpa.")

    print(f"\n{'='*65}")
    print(f"Batching Benchmark  |  model={model.name}  gpu={torch.cuda.get_device_name(0)}")
    print(f"seq_len={args.seq_len}  heads={model.num_heads}  head_dim={model.head_dim}  dtype={args.dtype}")
    print(f"{'='*65}")
    print("Speedup = batched tok/s / sequential tok/s  (higher is better)")

    for backend_name in backends:
        eff_dtype = dtype
        if backend_name == "flashattn_v0" and dtype != torch.float32:
            print(f"\n  [warn] flashattn_v0 requires fp32, ignoring --dtype {args.dtype}")
            eff_dtype = torch.float32
        try:
            run_batching_bench(
                backend_name, model, args.batch_sizes,
                args.seq_len, eff_dtype, args.num_warmup, args.num_runs, device,
            )
        except Exception as e:
            print(f"\n  {backend_name}: FAILED — {e}")

    print()


if __name__ == "__main__":
    main()
