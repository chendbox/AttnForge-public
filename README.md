# AttnForge

AttnForge is a research-oriented LLM inference project focused on the
attention path: prefill, decode, and KV cache management. The repository
contains a benchmarkable CUDA kernel playground plus the surrounding runtime,
correctness tests, and performance analysis tooling.

## Repository Layout

| Folder | Purpose |
|--------|---------|
| [`flashinfer_engine/`](flashinfer_engine/) | Benchmark harness, backend abstraction, CUDA kernels, tests, and documentation |

## Quick Start

```bash
# Activate environment
source .venv/bin/activate
export LD_LIBRARY_PATH=$(pwd)/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Build the available kernels
cd flashinfer_engine
python -m csrc.compile

# Run smoke tests
pytest tests/test_smoke.py -v

# Run benchmarks against the PyTorch SDPA baseline
python benchmarks/benchmark_prefill.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa flashattn_v0
python benchmarks/benchmark_decode.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa_naive flashattn_v0
python benchmarks/plot_results.py
```

## Project Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| **I0** | Benchmark harness with SDPA baseline + v0 kernel comparison | Done |
| **I1** | bf16 + Tensor Cores (`flashattn_v1`) | In progress |
| **I2** | GQA-native KV cache (`flashattn_v2`) | Planned |
| **I3** | Lightweight runtime integration | Planned |

See [`flashinfer_engine/docs/PERFORMANCE.md`](flashinfer_engine/docs/PERFORMANCE.md)
for the current benchmark snapshot and
[`flashinfer_engine/docs/PROJECT_ROADMAP.md`](flashinfer_engine/docs/PROJECT_ROADMAP.md)
for the next optimization stages.

## Hardware

Validated on NVIDIA H100 80GB with Llama-3.2-1B and Llama-3-8B model shapes.

## License

MIT - see [LICENSE](LICENSE).
