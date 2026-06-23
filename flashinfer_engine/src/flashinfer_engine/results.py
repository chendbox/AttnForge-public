from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


@dataclass
class BenchResult:
    # identity
    benchmark: str                  # "prefill" | "decode" | "e2e"
    model: str
    backend: str
    dtype: str

    # shape
    batch_size: int
    prompt_len: int
    output_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    # latency (ms)
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    # throughput
    tokens_per_sec: float

    # system
    vram_peak_gb: float
    gpu: str = field(default_factory=_gpu_name)
    git_commit: str = field(default_factory=_git_commit)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def save_result(result: BenchResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    fname = out / f"{result.benchmark}_{result.model}_{result.backend}_b{result.batch_size}_s{result.prompt_len}_{ts}.json"
    fname.write_text(json.dumps(asdict(result), indent=2))
    return fname


def load_results(results_dir: str | Path) -> list[BenchResult]:
    results = []
    for p in Path(results_dir).glob("*.json"):
        data = json.loads(p.read_text())
        results.append(BenchResult(**data))
    return results
