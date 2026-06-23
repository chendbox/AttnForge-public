# Benchmarks

Use this folder for standardized benchmark runners and result artifacts.

Recommended additions:

- `benchmark_prefill.py`
- `benchmark_decode.py`
- `results/*.json` with metadata (GPU, driver, CUDA, PyTorch, commit hash)

For now, baseline benchmark execution is driven by:

```bash
bash ../scripts/run_all.sh
```
