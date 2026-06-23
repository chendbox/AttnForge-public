#!/usr/bin/env python3
"""
Benchmark regression detector.

Compares two sets of JSON result files (e.g. from two git commits) and
prints a markdown table showing latency changes. Exits with code 1 if any
benchmark regressed beyond the threshold.

Usage:
    # Compare two result directories
    python benchmarks/compare_commits.py \\
        --baseline benchmarks/results/baseline/ \\
        --current  benchmarks/results/

    # Set regression threshold (default 5%)
    python benchmarks/compare_commits.py \\
        --baseline benchmarks/results/baseline/ \\
        --current  benchmarks/results/ \\
        --threshold 0.05

    # Output markdown to file
    python benchmarks/compare_commits.py \\
        --baseline benchmarks/results/baseline/ \\
        --current  benchmarks/results/ \\
        --out regression_report.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flashinfer_engine.results import BenchResult, load_results

_GREEN = "\033[32m"
_RED   = "\033[31m"
_RESET = "\033[0m"


def _key(r: BenchResult) -> str:
    return f"{r.benchmark}/{r.model}/{r.backend}/b{r.batch_size}_s{r.prompt_len}"


def _pct(baseline: float, current: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline * 100.0


def _arrow(pct: float, threshold_pct: float) -> str:
    if abs(pct) < 1.0:
        return "→"
    if pct > threshold_pct:
        return "▲ REGRESSED"
    if pct < -threshold_pct:
        return "▼ improved"
    return "↑" if pct > 0 else "↓"


def compare(
    baseline_dir: Path,
    current_dir: Path,
    threshold: float = 0.05,
    out_path: Path | None = None,
) -> bool:
    baseline_results = {_key(r): r for r in load_results(baseline_dir)}
    current_results  = {_key(r): r for r in load_results(current_dir)}

    if not baseline_results:
        print(f"No baseline results found in {baseline_dir}", file=sys.stderr)
        sys.exit(1)
    if not current_results:
        print(f"No current results found in {current_dir}", file=sys.stderr)
        sys.exit(1)

    threshold_pct = threshold * 100.0
    rows: list[dict] = []
    any_regression = False

    for key in sorted(set(baseline_results) | set(current_results)):
        if key not in baseline_results:
            rows.append({"key": key, "status": "NEW", "base": "-", "cur": "-", "delta": "-"})
            continue
        if key not in current_results:
            rows.append({"key": key, "status": "MISSING", "base": "-", "cur": "-", "delta": "-"})
            continue

        base = baseline_results[key]
        cur  = current_results[key]
        pct  = _pct(base.p50_ms, cur.p50_ms)
        arrow = _arrow(pct, threshold_pct)

        if "REGRESSED" in arrow:
            any_regression = True

        rows.append({
            "key":    key,
            "status": arrow,
            "base":   f"{base.p50_ms:.3f} ms",
            "cur":    f"{cur.p50_ms:.3f} ms",
            "delta":  f"{pct:+.1f}%",
            "base_tok": f"{base.tokens_per_sec:,.0f}",
            "cur_tok":  f"{cur.tokens_per_sec:,.0f}",
        })

    # Build markdown
    lines = [
        "## Benchmark Regression Report\n",
        f"Baseline: `{baseline_dir}`  ",
        f"Current:  `{current_dir}`  ",
        f"Threshold: {threshold_pct:.0f}%\n",
        "| Benchmark | Baseline p50 | Current p50 | Δ | Status |",
        "|-----------|-------------|------------|---|--------|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['key']}` | {r['base']} | {r['cur']} | {r['delta']} | {r['status']} |"
        )

    if any_regression:
        lines.append(f"\n> ⚠️ **Regressions detected** (threshold: {threshold_pct:.0f}%)")
    else:
        lines.append(f"\n> ✅ No regressions above {threshold_pct:.0f}% threshold")

    md = "\n".join(lines)
    print(md)

    if out_path:
        out_path.write_text(md)
        print(f"\nSaved → {out_path}", file=sys.stderr)

    return any_regression


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark regression detector")
    parser.add_argument("--baseline", required=True, type=Path,
                        help="Directory with baseline JSON results")
    parser.add_argument("--current", required=True, type=Path,
                        help="Directory with current JSON results")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Regression threshold as fraction (default: 0.05 = 5%%)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write markdown report to this file")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit with code 1 if any regression detected")
    args = parser.parse_args()

    regressed = compare(args.baseline, args.current, args.threshold, args.out)

    if regressed and args.fail_on_regression:
        sys.exit(1)


if __name__ == "__main__":
    main()
