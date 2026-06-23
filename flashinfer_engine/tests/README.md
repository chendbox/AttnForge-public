# Tests

Integration and regression tests for the inference engine. Run with `pytest`.

## Test files

| File | Tag | What it covers |
|------|-----|----------------|
| `test_smoke.py`       | CPU-only | Package imports, config loading, result schema, metric utilities |
| `test_correctness.py` | GPU      | Every prefill backend agrees with PyTorch SDPA within fp32 tolerance; causal mask actually masks |
| `test_kv_cache.py`    | GPU      | Decode backends match SDPA; `update_kv_cache` writes correct position; incremental decode == full recompute |

GPU tests are auto-skipped when `torch.cuda.is_available()` is False, so
`test_smoke.py` always runs in CI.

## Running

```bash
# All tests
pytest

# CPU-only (skip GPU tests)
pytest -m "not gpu"

# Just smoke tests
pytest tests/test_smoke.py

# Verbose with full diff on failures
pytest -vv

# Specific test
pytest tests/test_correctness.py::test_prefill_matches_sdpa

# Coverage
pytest --cov=flashinfer_engine --cov-report=term-missing
```

## Prerequisites for GPU tests

1. CUDA GPU available
2. v0 kernels compiled (`python -m csrc.compile`)
3. `LD_LIBRARY_PATH` includes `.venv/lib/python3.12/site-packages/torch/lib`

If the v0 kernels aren't compiled, the `flashattn_v0` cases are skipped and
only SDPA-vs-SDPA correctness runs (a sanity check for the test harness itself).

## Kernel build correctness gate

From the engine root:

| Step | Command | Validates |
|------|---------|-----------|
| 1 | `python -m csrc.compile --prefill-only --v0-only` | flashattn_v0 prefill kernel builds |
| 2 | `python -m csrc.compile --decode-only --v0-only`  | flashattn_v0 decode kernel builds |
| 3 | `pytest tests/test_correctness.py -v`             | flashattn_v0 prefill numerical correctness |
| 4 | `pytest tests/test_kv_cache.py -v`                | flashattn_v0 decode + KV cache correctness |

Or run the smoke path from the engine root via `bash scripts/run_all.sh`.

## Planned

- `test_shapes.py` - explicit edge case sweep (batch=1 seq=1, odd head_dims)
- `test_dtype.py` - when v1 lands, parametrize over fp32/bf16/fp16 with
  per-dtype tolerance buckets
- benchmark threshold alerts (regression detection on `benchmarks/results/*.json`)
- multi-GPU compatibility matrix
