#!/usr/bin/env python3
"""
Decode attention benchmark.

Measures single-token decode latency (TBT) against a KV cache of increasing
context length. Compares naive SDPA re-computation vs. our v0 decode kernel.

Usage:
    python benchmarks/benchmark_decode.py --model configs/models/llama3_2_1b.yaml
    python benchmarks/benchmark_decode.py \\
        --model configs/models/llama3_8b.yaml \\
        --backends sdpa_naive flashattn_v0 \\
        --context-lens 512 1024 2048 4096 \\
        --batch-sizes 1 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.backends import available_decode_backends, get_decode_backend
from flashinfer_engine.config import ModelConfig
from flashinfer_engine.metrics import measure_latency, percentile_stats, reset_vram_peak, vram_peak_gb
from flashinfer_engine.results import BenchResult, save_result

_CORRECTNESS_RTOL = 1e-3
_CORRECTNESS_ATOL = 1e-2


def _check_decode_correctness(
    backend_name: str, backend, q, k_cache, v_cache, context_len: int,
    num_heads: int, head_dim: int,
) -> bool:
    """Compare decode backend against SDPA reference. Returns True if correct."""
    if backend_name == "sdpa_naive":
        return True  # sdpa_naive is the reference
    with torch.no_grad():
        # Build SDPA reference from the same cache data
        batch = q.shape[0]
        # k_cache is (batch, max_len, hidden) for v0; reshape to (batch, heads, ctx, head_dim)
        k_ref = (k_cache[:, :context_len, :]
                 .view(batch, context_len, num_heads, head_dim)
                 .transpose(1, 2).contiguous())
        v_ref = (v_cache[:, :context_len, :]
                 .view(batch, context_len, num_heads, head_dim)
                 .transpose(1, 2).contiguous())
        ref = F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=False)
        out = backend.decode(q, k_cache, v_cache, context_len)
    ok = torch.allclose(out, ref, rtol=_CORRECTNESS_RTOL, atol=_CORRECTNESS_ATOL)
    if ok:
        print(f"  [correctness] PASS  (max_diff={(out - ref).abs().max().item():.2e})")
    else:
        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        print(f"  [correctness] FAIL  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
              f"  rtol={_CORRECTNESS_RTOL}  atol={_CORRECTNESS_ATOL}")
    return ok

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _alloc_decode_tensors(
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    context_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    v0_decode_format: bool = False,
):
    """Allocate Q (single token) and KV cache filled with random context."""
    q = torch.randn(batch, num_q_heads, 1, head_dim, dtype=dtype, device=device)

    if v0_decode_format:
        # v0 decode kernel is MHA-only: K/V hidden must match Q hidden (= num_q_heads * head_dim).
        # Native GQA cache support is I3 scope.
        hidden_dim = num_q_heads * head_dim
        max_len = context_len + 64  # small headroom
        k_cache = torch.randn(batch, max_len, hidden_dim, dtype=dtype, device=device)
        v_cache = torch.randn(batch, max_len, hidden_dim, dtype=dtype, device=device)
    else:
        # SDPA expects (batch, heads, seq, head_dim)
        k_cache = torch.randn(batch, num_kv_heads, context_len, head_dim, dtype=dtype, device=device)
        v_cache = torch.randn(batch, num_kv_heads, context_len, head_dim, dtype=dtype, device=device)

    return q, k_cache, v_cache


def run_one(
    backend_name: str,
    model: ModelConfig,
    batch: int,
    context_len: int,
    dtype_str: str,
    num_warmup: int,
    num_runs: int,
    output_dir: str,
    device: torch.device,
    check_correctness: bool = False,
) -> BenchResult | None:
    backend = get_decode_backend(backend_name)
    dtype = _DTYPE_MAP[dtype_str]

    if backend_name == "flashattn_v0" and dtype != torch.float32:
        print(f"  [warn] v0 decode kernel requires fp32, ignoring --dtype {dtype_str}")
        dtype = torch.float32

    v0_decode_format = backend_name == "flashattn_v0"
    q, k_cache, v_cache = _alloc_decode_tensors(
        batch, model.num_heads, model.num_kv_heads,
        context_len, model.head_dim, dtype, device,
        v0_decode_format=v0_decode_format,
    )

    # SDPA: expand KV if GQA
    if not v0_decode_format and model.is_gqa:
        k_cache = k_cache.repeat_interleave(model.gqa_groups, dim=1)
        v_cache = v_cache.repeat_interleave(model.gqa_groups, dim=1)

    if check_correctness and v0_decode_format:
        if not _check_decode_correctness(
            backend_name, backend, q, k_cache, v_cache, context_len,
            model.num_heads, model.head_dim,
        ):
            return None

    reset_vram_peak()

    def fn():
        with torch.no_grad():
            backend.decode(q, k_cache, v_cache, context_len)

    times = measure_latency(fn, num_warmup=num_warmup, num_runs=num_runs)
    stats = percentile_stats(times)

    # decode throughput: 1 token per request per step
    tokens_per_sec = batch / (stats["p50_ms"] / 1000.0)
    peak_gb = vram_peak_gb()

    result = BenchResult(
        benchmark="decode",
        model=model.name,
        backend=backend_name,
        dtype=dtype_str,
        batch_size=batch,
        prompt_len=context_len,   # context_len stored in prompt_len field
        output_len=1,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        tokens_per_sec=tokens_per_sec,
        vram_peak_gb=peak_gb,
        **stats,
    )

    path = save_result(result, output_dir)
    print(
        f"  {backend_name:16s} | batch={batch} ctx={context_len:5d} | "
        f"p50={stats['p50_ms']:7.3f}ms  p95={stats['p95_ms']:7.3f}ms  "
        f"p99={stats['p99_ms']:7.3f}ms  {tokens_per_sec:8.1f} tok/s  "
        f"vram={peak_gb:.2f}GB"
    )
    print(f"  -> saved {path.name}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Decode attention benchmark (TBT)")
    parser.add_argument("--model", required=True, help="Path to model yaml config")
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="Backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--context-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096],
        help="KV cache context lengths to sweep",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results")
    parser.add_argument(
        "--check-correctness", action="store_true",
        help="Compare each backend against SDPA before timing. Fails the run on mismatch.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for this benchmark.")
        sys.exit(1)

    device = torch.device("cuda")
    model = ModelConfig.from_yaml(args.model)

    backends = args.backends or available_decode_backends()
    if not backends:
        print("No decode backends available. Compile v0 kernels or use 'sdpa_naive'.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Decode Benchmark (TBT)  |  model={model.name}  gpu={torch.cuda.get_device_name(0)}")
    print(f"heads={model.num_heads}  kv_heads={model.num_kv_heads}  head_dim={model.head_dim}")
    print(f"{'='*70}")

    for batch in args.batch_sizes:
        for ctx_len in args.context_lens:
            print(f"\nbatch={batch}  context_len={ctx_len}")
            for backend_name in backends:
                try:
                    run_one(
                        backend_name, model, batch, ctx_len,
                        args.dtype, args.num_warmup, args.num_runs,
                        args.out, device,
                        check_correctness=args.check_correctness,
                    )
                except Exception as e:
                    print(f"  {backend_name:16s} | FAILED: {e}")

    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()
