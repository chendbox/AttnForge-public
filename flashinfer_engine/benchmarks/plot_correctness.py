#!/usr/bin/env python3
"""
Correctness comparison plot.

For each sequence length, measures max absolute difference between each
backend and the SDPA reference. Saves the figure to docs/figures/.

Usage:
    python benchmarks/plot_correctness.py --model configs/models/llama3_2_1b.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.backends import get_prefill_backend, get_decode_backend
from flashinfer_engine.config import ModelConfig

ATOL = 1e-2  # pass threshold shown as horizontal line

SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096]


# ---------- collectors ----------

def collect_prefill(model: ModelConfig, seq_lens: list[int], device: torch.device):
    """Returns dict: backend_name -> list of (max_diff, mean_diff) per seq_len."""
    results: dict[str, list[tuple[float, float]]] = {"flashattn_v0": []}

    backend = get_prefill_backend("flashattn_v0")
    if not backend.available():
        print("[prefill] flashattn_v0 not available, skipping.")
        return {}

    for seq_len in seq_lens:
        q = torch.randn(1, model.num_heads, seq_len, model.head_dim,
                        dtype=torch.float32, device=device)
        k = torch.randn(1, model.num_heads, seq_len, model.head_dim,
                        dtype=torch.float32, device=device)
        v = torch.randn(1, model.num_heads, seq_len, model.head_dim,
                        dtype=torch.float32, device=device)

        with torch.no_grad():
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = backend.prefill(q, k, v, causal=True)

        diff = (out - ref).abs()
        results["flashattn_v0"].append((diff.max().item(), diff.mean().item()))
        print(f"  prefill  seq={seq_len:5d}  max_diff={diff.max().item():.2e}  "
              f"mean_diff={diff.mean().item():.2e}")

    return results


def collect_decode(model: ModelConfig, seq_lens: list[int], device: torch.device):
    """Returns dict: backend_name -> list of (max_diff, mean_diff) per seq_len."""
    results: dict[str, list[tuple[float, float]]] = {"flashattn_v0": []}

    backend = get_decode_backend("flashattn_v0")
    if not backend.available():
        print("[decode] flashattn_v0 not available, skipping.")
        return {}

    hidden = model.num_heads * model.head_dim

    for ctx_len in seq_lens:
        q = torch.randn(1, model.num_heads, 1, model.head_dim,
                        dtype=torch.float32, device=device)
        k_cache = torch.randn(1, ctx_len + 32, hidden,
                              dtype=torch.float32, device=device)
        v_cache = torch.randn(1, ctx_len + 32, hidden,
                              dtype=torch.float32, device=device)

        k_ref = (k_cache[:, :ctx_len, :]
                 .view(1, ctx_len, model.num_heads, model.head_dim)
                 .transpose(1, 2).contiguous())
        v_ref = (v_cache[:, :ctx_len, :]
                 .view(1, ctx_len, model.num_heads, model.head_dim)
                 .transpose(1, 2).contiguous())

        with torch.no_grad():
            ref = F.scaled_dot_product_attention(q, k_ref, v_ref, is_causal=False)
            out = backend.decode(q, k_cache, v_cache, ctx_len)

        diff = (out - ref).abs()
        results["flashattn_v0"].append((diff.max().item(), diff.mean().item()))
        print(f"  decode   ctx={ctx_len:5d}  max_diff={diff.max().item():.2e}  "
              f"mean_diff={diff.mean().item():.2e}")

    return results


# ---------- plot ----------

def plot(
    prefill_results: dict,
    decode_results: dict,
    seq_lens: list[int],
    out_path: Path,
    gpu_name: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Correctness: flashattn_v0 vs SDPA reference\n"
        f"fp32 · batch=1 · {gpu_name}",
        fontsize=13, fontweight="bold",
    )

    colors = {"max_diff": "#e05252", "mean_diff": "#5285e0"}

    for ax, results, title, xlabel in [
        (axes[0], prefill_results, "Prefill  (causal=True)", "Sequence length"),
        (axes[1], decode_results,  "Decode   (context length)", "Context length"),
    ]:
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absolute difference")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(seq_lens)

        if not results or "flashattn_v0" not in results:
            ax.text(0.5, 0.5, "kernel not available", transform=ax.transAxes,
                    ha="center", va="center", color="gray")
            continue

        data = results["flashattn_v0"]
        max_diffs  = [d[0] for d in data]
        mean_diffs = [d[1] for d in data]

        ax.plot(seq_lens, max_diffs,  "o-", color=colors["max_diff"],
                linewidth=2, markersize=6, label="max |diff|")
        ax.plot(seq_lens, mean_diffs, "s--", color=colors["mean_diff"],
                linewidth=1.5, markersize=5, label="mean |diff|")

        # pass threshold
        ax.axhline(ATOL, color="gray", linestyle=":", linewidth=1.2,
                   label=f"atol threshold ({ATOL:.0e})")

        # fp32 machine epsilon reference
        ax.axhline(1.2e-7, color="#aaaaaa", linestyle="-.", linewidth=0.8,
                   label="fp32 machine ε (1.2e-7)")

        # PASS / FAIL annotations
        for i, (sl, md) in enumerate(zip(seq_lens, max_diffs)):
            label = "PASS" if md < ATOL else "FAIL"
            color = "#2a9d2a" if md < ATOL else "#cc0000"
            ax.annotate(label, (sl, md), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7,
                        color=color, fontweight="bold")

        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)

        # y-axis range: from just below fp32 ε to just above atol
        ax.set_ylim(5e-8, 1.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--seq-lens", nargs="+", type=int, default=SEQ_LENS,
        help="Sequence / context lengths to sweep",
    )
    parser.add_argument("--out", default="docs/figures/correctness.png")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required.")
        sys.exit(1)

    device = torch.device("cuda")
    model = ModelConfig.from_yaml(args.model)
    gpu_name = torch.cuda.get_device_name(0)

    print(f"Model: {model.name}  GPU: {gpu_name}")
    print(f"Sweeping seq_lens: {args.seq_lens}\n")

    print("[Prefill]")
    prefill_res = collect_prefill(model, args.seq_lens, device)

    print("\n[Decode]")
    decode_res = collect_decode(model, args.seq_lens, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot(prefill_res, decode_res, args.seq_lens, out_path, gpu_name)


if __name__ == "__main__":
    main()
