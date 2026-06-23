from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch


def measure_latency(
    fn: Callable,
    num_warmup: int = 5,
    num_runs: int = 20,
    cuda: bool = True,
) -> list[float]:
    """Returns per-run latencies in milliseconds."""
    sync = torch.cuda.synchronize if cuda and torch.cuda.is_available() else lambda: None

    for _ in range(num_warmup):
        fn()
    sync()

    times: list[float] = []
    for _ in range(num_runs):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    return times


def percentile_stats(times: list[float]) -> dict[str, float]:
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
