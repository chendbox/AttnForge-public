"""CPU-only smoke tests.

These should always pass on any machine — they verify the package imports
cleanly, configs parse, and the result schema is well-formed. Designed to
run in CI without a GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashinfer_engine import (
    BenchResult,
    ModelConfig,
    load_results,
    measure_latency,
    percentile_stats,
    save_result,
)
from flashinfer_engine.backends import (
    _DECODE_BACKENDS,
    _PREFILL_BACKENDS,
    available_decode_backends,
    available_prefill_backends,
)


CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "models"


# ---------- imports & package metadata ----------

def test_package_version():
    import flashinfer_engine as fe
    assert hasattr(fe, "__version__")
    assert isinstance(fe.__version__, str)


def test_backend_registries_present():
    """SDPA must always be registered (no GPU needed for registration)."""
    assert "sdpa" in _PREFILL_BACKENDS
    assert "sdpa_naive" in _DECODE_BACKENDS
    assert "flashattn_v0" in _PREFILL_BACKENDS
    assert "flashattn_v0" in _DECODE_BACKENDS


def test_available_backends_returns_list():
    # at least sdpa should always be available (it's pure PyTorch)
    assert "sdpa" in available_prefill_backends()
    assert "sdpa_naive" in available_decode_backends()


# ---------- model config loading ----------

@pytest.mark.parametrize("yaml_name", ["llama3_2_1b.yaml", "llama3_8b.yaml"])
def test_model_config_loads(yaml_name):
    cfg = ModelConfig.from_yaml(CONFIGS_DIR / yaml_name)
    assert cfg.hidden_dim > 0
    assert cfg.num_heads > 0
    assert cfg.num_kv_heads > 0
    assert cfg.head_dim == cfg.hidden_dim // cfg.num_heads


def test_llama3_2_1b_is_gqa():
    cfg = ModelConfig.from_yaml(CONFIGS_DIR / "llama3_2_1b.yaml")
    assert cfg.is_gqa is True
    assert cfg.gqa_groups == 4   # 32 Q heads / 8 KV heads
    assert cfg.kv_hidden_dim == cfg.num_kv_heads * cfg.head_dim


def test_llama3_8b_shape():
    cfg = ModelConfig.from_yaml(CONFIGS_DIR / "llama3_8b.yaml")
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.hidden_dim == 4096


# ---------- result schema ----------

def _make_dummy_result() -> BenchResult:
    return BenchResult(
        benchmark="prefill",
        model="llama3_2_1b",
        backend="sdpa",
        dtype="fp32",
        batch_size=1,
        prompt_len=512,
        output_len=0,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        mean_ms=0.1,
        p50_ms=0.1,
        p95_ms=0.11,
        p99_ms=0.12,
        min_ms=0.09,
        max_ms=0.13,
        tokens_per_sec=5_000_000.0,
        vram_peak_gb=0.02,
    )


def test_bench_result_dataclass_construction():
    r = _make_dummy_result()
    assert r.benchmark == "prefill"
    assert r.gpu is not None     # default_factory should populate
    assert r.git_commit is not None
    assert r.timestamp is not None


def test_save_and_load_roundtrip(tmp_path):
    r = _make_dummy_result()
    path = save_result(r, tmp_path)
    assert path.exists()

    loaded = load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].backend == "sdpa"
    assert loaded[0].p50_ms == pytest.approx(0.1)


def test_save_result_json_is_valid(tmp_path):
    r = _make_dummy_result()
    path = save_result(r, tmp_path)
    data = json.loads(path.read_text())
    # Required fields present
    for field in [
        "benchmark", "model", "backend", "dtype",
        "batch_size", "prompt_len", "output_len",
        "p50_ms", "p95_ms", "p99_ms", "tokens_per_sec",
        "gpu", "git_commit", "timestamp",
    ]:
        assert field in data, f"Missing field {field} in saved JSON"


# ---------- metrics utilities ----------

def test_percentile_stats_basic():
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = percentile_stats(times)
    assert stats["min_ms"] == 1.0
    assert stats["max_ms"] == 5.0
    assert stats["p50_ms"] == pytest.approx(3.0)
    # mean
    assert stats["mean_ms"] == pytest.approx(3.0)


def test_measure_latency_cpu():
    # measure a trivial CPU op — just verify the harness works without CUDA
    times = measure_latency(
        lambda: sum(range(100)),
        num_warmup=2,
        num_runs=5,
        cuda=False,
    )
    assert len(times) == 5
    assert all(t >= 0 for t in times)
