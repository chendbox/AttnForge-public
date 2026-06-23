#!/usr/bin/env python3
"""
Prefill attention benchmark.

Measures the latency of the attention kernel alone (no linear projections, no MLP)
across backends, sequence lengths, and batch sizes.

Usage:
    python benchmarks/benchmark_prefill.py --model configs/models/llama3_2_1b.yaml
    python benchmarks/benchmark_prefill.py \\
        --model configs/models/llama3_8b.yaml \\
        --backends sdpa flashattn_v0 \\
        --prompt-lens 512 1024 2048 4096 \\
        --batch-sizes 1 4 \\
        --dtype fp32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# allow running from repo root or from flashinfer_engine/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.backends import available_prefill_backends, get_prefill_backend
from flashinfer_engine.config import ModelConfig
from flashinfer_engine.metrics import measure_latency, percentile_stats, reset_vram_peak, vram_peak_gb
from flashinfer_engine.results import BenchResult, save_result

_CORRECTNESS_RTOL = 1e-3
_CORRECTNESS_ATOL = 1e-2


def _correctness_tolerances(backend_name: str) -> tuple[float, float]:
    """v1 runs through bf16 internally, so it needs bf16-appropriate tolerances."""
    if backend_name == "flashattn_v1":
        return 1e-2, 1e-2
    return _CORRECTNESS_RTOL, _CORRECTNESS_ATOL


def _check_prefill_correctness(backend_name: str, backend, q, k, v) -> bool:
    """Compare backend output against SDPA reference. Returns True if correct."""
    if backend_name == "sdpa":
        return True  # sdpa is the reference
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = backend.prefill(q, k, v, causal=True)
    rtol, atol = _correctness_tolerances(backend_name)
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    if ok:
        print(f"  [correctness] PASS  (max_diff={( out - ref).abs().max().item():.2e})")
    else:
        max_diff = (out - ref).abs().max().item()
        mean_diff = (out - ref).abs().mean().item()
        print(f"  [correctness] FAIL  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
              f"  rtol={rtol}  atol={atol}")
    return ok

_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _alloc_qkv(
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
):
    q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def run_one(
    backend_name: str,
    model: ModelConfig,
    batch: int,
    prompt_len: int,
    dtype_str: str,
    num_warmup: int,
    num_runs: int,
    output_dir: str,
    device: torch.device,
    check_correctness: bool = False,
) -> BenchResult | None:
    backend = get_prefill_backend(backend_name)
    dtype = _DTYPE_MAP[dtype_str]

    # v0 prefill kernel is fp32-only; warn if mismatch
    if backend_name == "flashattn_v0" and dtype != torch.float32:
        print(f"  [warn] v0 prefill kernel requires fp32, ignoring --dtype {dtype_str}")
        dtype = torch.float32

    q, k, v = _alloc_qkv(
        batch, model.num_heads, model.num_kv_heads,
        prompt_len, model.head_dim, dtype, device,
    )

    # All current backends are MHA-only: expand KV heads to match Q heads for GQA models.
    # (v0 prefill kernel assumes num_kv_heads == num_q_heads; native GQA support is I3 scope.)
    if model.is_gqa:
        k = k.repeat_interleave(model.gqa_groups, dim=1)
        v = v.repeat_interleave(model.gqa_groups, dim=1)

    if check_correctness:
        if not _check_prefill_correctness(backend_name, backend, q, k, v):
            return None

    reset_vram_peak()

    def fn():
        with torch.no_grad():
            backend.prefill(q, k, v, causal=True)

    times = measure_latency(fn, num_warmup=num_warmup, num_runs=num_runs)
    stats = percentile_stats(times)

    tokens_per_sec = (batch * prompt_len) / (stats["p50_ms"] / 1000.0)
    peak_gb = vram_peak_gb()

    result = BenchResult(
        benchmark="prefill",
        model=model.name,
        backend=backend_name,
        dtype=dtype_str,
        batch_size=batch,
        prompt_len=prompt_len,
        output_len=0,
        num_heads=model.num_heads,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        tokens_per_sec=tokens_per_sec,
        vram_peak_gb=peak_gb,
        **stats,
    )

    path = save_result(result, output_dir)
    print(
        f"  {backend_name:16s} | batch={batch} seq={prompt_len:5d} | "
        f"p50={stats['p50_ms']:7.2f}ms  p95={stats['p95_ms']:7.2f}ms  "
        f"p99={stats['p99_ms']:7.2f}ms  {tokens_per_sec:10.0f} tok/s  "
        f"vram={peak_gb:.2f}GB"
    )
    print(f"  -> saved {path.name}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Prefill attention benchmark")
    parser.add_argument("--model", required=True, help="Path to model yaml config")
    parser.add_argument(
        "--backends", nargs="+", default=None,
        help="Backends to benchmark (default: all available)",
    )
    parser.add_argument(
        "--prompt-lens", nargs="+", type=int, default=[512, 1024, 2048, 4096],
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="fp32",
        help="Tensor dtype (v0 prefill kernel forces fp32)",
    )
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--out", default="benchmarks/results", help="Output directory")
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

    backends = args.backends or available_prefill_backends()
    if not backends:
        print("No prefill backends available. Compile v0 kernels or use 'sdpa'.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Prefill Benchmark  |  model={model.name}  gpu={torch.cuda.get_device_name(0)}")
    print(f"heads={model.num_heads}  kv_heads={model.num_kv_heads}  head_dim={model.head_dim}")
    print(f"{'='*70}")

    for batch in args.batch_sizes:
        for prompt_len in args.prompt_lens:
            print(f"\nbatch={batch}  prompt_len={prompt_len}")
            for backend_name in backends:
                try:
                    run_one(
                        backend_name, model, batch, prompt_len,
                        args.dtype, args.num_warmup, args.num_runs,
                        args.out, device,
                        check_correctness=args.check_correctness,
                    )
                except Exception as e:
                    print(f"  {backend_name:16s} | FAILED: {e}")

    print(f"\nDone. Results in {args.out}/")


if __name__ == "__main__":
    main()
