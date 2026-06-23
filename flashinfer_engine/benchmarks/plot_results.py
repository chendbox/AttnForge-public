#!/usr/bin/env python3
"""
Plot benchmark results from benchmarks/results/*.json.

Produces:
  docs/figures/prefill_latency.png   — p50 latency vs seq_len per backend
  docs/figures/decode_tbt.png        — TBT vs context_len per backend
  docs/figures/speedup.png           — speedup of each backend over sdpa baseline

Usage:
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --results-dir benchmarks/results --out docs/figures
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.results import load_results, BenchResult


def _group_by(results: list[BenchResult], keys: list[str]) -> dict:
    groups: dict = defaultdict(list)
    for r in results:
        k = tuple(getattr(r, key) for key in keys)
        groups[k].append(r)
    return groups


def plot_prefill_latency(results: list[BenchResult], out_dir: Path) -> None:
    data = [r for r in results if r.benchmark == "prefill"]
    if not data:
        print("No prefill results found, skipping prefill plot.")
        return

    backends = sorted({r.backend for r in data})
    batch_sizes = sorted({r.batch_size for r in data})

    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(6 * len(batch_sizes), 5), squeeze=False)

    for col, batch in enumerate(batch_sizes):
        ax = axes[0][col]
        for backend in backends:
            pts = sorted(
                [r for r in data if r.batch_size == batch and r.backend == backend],
                key=lambda r: r.prompt_len,
            )
            if not pts:
                continue
            xs = [r.prompt_len for r in pts]
            ys = [r.p50_ms for r in pts]
            errs = [[r.p50_ms - r.min_ms for r in pts], [r.p99_ms - r.p50_ms for r in pts]]
            ax.errorbar(xs, ys, yerr=errs, marker="o", label=backend, capsize=4)

        ax.set_title(f"Prefill latency — batch={batch}")
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel("p50 latency (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "prefill_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_decode_tbt(results: list[BenchResult], out_dir: Path) -> None:
    data = [r for r in results if r.benchmark == "decode"]
    if not data:
        print("No decode results found, skipping TBT plot.")
        return

    backends = sorted({r.backend for r in data})
    batch_sizes = sorted({r.batch_size for r in data})

    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(6 * len(batch_sizes), 5), squeeze=False)

    for col, batch in enumerate(batch_sizes):
        ax = axes[0][col]
        for backend in backends:
            pts = sorted(
                [r for r in data if r.batch_size == batch and r.backend == backend],
                key=lambda r: r.prompt_len,  # prompt_len stores context_len for decode
            )
            if not pts:
                continue
            xs = [r.prompt_len for r in pts]
            ys = [r.p50_ms for r in pts]
            ax.plot(xs, ys, marker="o", label=backend)

        ax.set_title(f"Decode TBT — batch={batch}")
        ax.set_xlabel("Context length (tokens)")
        ax.set_ylabel("Time between tokens p50 (ms)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "decode_tbt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_speedup(results: list[BenchResult], out_dir: Path, baseline: str = "sdpa") -> None:
    """Speedup of each backend over baseline across (batch, seq_len) sweep.

    For each benchmark type, produces one subplot per batch_size, with bars
    for each backend at every seq_len.  A dashed line at y=1 marks parity.
    """
    all_benchmarks = sorted({r.benchmark for r in results})
    for bench in all_benchmarks:
        data = [r for r in results if r.benchmark == bench]
        base_name = baseline if bench == "prefill" else f"{baseline}_naive"

        baselines = [r for r in data if r.backend == base_name]
        others = [r for r in data if r.backend != base_name]
        if not baselines or not others:
            continue

        # index baseline by (batch, seq) for O(1) lookup
        base_idx = {(r.batch_size, r.prompt_len): r for r in baselines}

        batch_sizes = sorted({r.batch_size for r in data})
        seq_lens = sorted({r.prompt_len for r in data})
        backends = sorted({r.backend for r in others})

        fig, axes = plt.subplots(
            1, len(batch_sizes), figsize=(6 * len(batch_sizes), 5), squeeze=False, sharey=True,
        )

        x = np.arange(len(seq_lens))
        width = 0.8 / max(len(backends), 1)

        for col, batch in enumerate(batch_sizes):
            ax = axes[0][col]
            for i, backend in enumerate(backends):
                speedups = []
                for seq in seq_lens:
                    base = base_idx.get((batch, seq))
                    other = next(
                        (r for r in others
                         if r.batch_size == batch and r.prompt_len == seq and r.backend == backend),
                        None,
                    )
                    if base and other and other.p50_ms > 0:
                        speedups.append(base.p50_ms / other.p50_ms)
                    else:
                        speedups.append(0.0)
                ax.bar(x + i * width, speedups, width, label=backend)

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="parity (=sdpa)")
            ax.set_xticks(x + width * (len(backends) - 1) / 2)
            ax.set_xticklabels([str(s) for s in seq_lens])
            ax.set_xlabel("seq_len" if bench == "prefill" else "context_len")
            if col == 0:
                ax.set_ylabel(f"Speedup over {base_name}  (>1 = faster)")
            ax.set_title(f"{bench.capitalize()} — batch={batch}")
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"{bench.capitalize()} speedup (p50 latency)", y=1.02)
        fig.tight_layout()
        path = out_dir / f"{bench}_speedup.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--results-dir", default="benchmarks/results")
    parser.add_argument("--out", default="docs/figures")
    parser.add_argument("--baseline", default="sdpa", help="Baseline backend name for speedup plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} result(s) from {args.results_dir}")
    plot_prefill_latency(results, out_dir)
    plot_decode_tbt(results, out_dir)
    plot_speedup(results, out_dir, baseline=args.baseline)
    print("Done.")


if __name__ == "__main__":
    main()
