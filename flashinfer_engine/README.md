# LLM Inference Engine

A research-oriented LLM inference runtime focused on the attention path:
prefill, decode, and KV cache management — measured against PyTorch SDPA
(cuDNN FlashAttention-2) on real model shapes (Llama-3.2-1B, Llama-3-8B).

## Architecture

```mermaid
flowchart LR
    User["User / CLI"] --> Pipeline["BenchPipeline"]
    Pipeline --> Dispatcher["BackendDispatcher"]
    Dispatcher --> P1["SDPAPrefill\n(cuDNN FA-2)"]
    Dispatcher --> P2["FlashV0Prefill\n(FA-1, fp32)"]
    Dispatcher --> P3["FlashV1Prefill\n(bf16+wmma, prefill-only)"]
    Dispatcher --> D1["SDPANaive\n(decode)"]
    Dispatcher --> D2["FlashV0Decode\n(FA-1, fp32)"]
    Dispatcher --> D3["FlashV2Decode\n(GQA-native, fp32)"]
    D2 <--> KV["KV Cache\n(batch, seq, hidden)"]
    D3 <--> GQAKV["GQA KV Cache\n(batch, kv_heads, seq, head_dim)"]
    Pipeline --> Results["BenchResult → JSON"]
```

→ Full diagram and component descriptions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| **I0** | Benchmark harness (prefill/decode, JSON results, plots) | Done |
| **I1** | `v1`: bf16 + Tensor Core prefill path | Done |
| **I2** | `v2`: GQA-native KV cache for decode | Done |
| **I3** | lightweight runtime integration for demos / MLIS | Planned |

## Kernel versions

| Backend | Description |
|---------|-------------|
| `sdpa` / `sdpa_naive` | PyTorch baseline (cuDNN FlashAttention-2 internally) |
| `flashattn_v0` | Custom kernel - fp32, MHA, tiled FlashAttention-1 style |
| `flashattn_v1` | bf16 + Tensor Cores (wmma), prefill-only |
| `flashattn_v2` | GQA-native fp32 single-token decode with static KV cache |
| `runtime` *(planned)* | lightweight inference integration over the optimized backends |

## Project Structure

```
src/flashinfer_engine/
  backends/             # attention backend implementations
  config.py             # model + benchmark config loaders
  metrics.py            # latency / percentile utilities
  results.py            # BenchResult schema + JSON I/O

benchmarks/             # benchmark drivers
  benchmark_prefill.py
  benchmark_decode.py
  plot_results.py
  results/              # *.json artifacts
configs/
  models/               # llama3_2_1b.yaml, llama3_8b.yaml
docs/                   # architecture notes + figures
tests/                  # correctness + regression tests
```

## Quick Start

```bash
# Install
cd flashinfer_engine
uv pip install -e .   # or: pip install -e .

# Run baseline benchmarks (SDPA only — works on any GPU)
python benchmarks/benchmark_prefill.py --model configs/models/llama3_2_1b.yaml
python benchmarks/benchmark_decode.py  --model configs/models/llama3_2_1b.yaml

# Compare against custom v0 kernel
python benchmarks/benchmark_prefill.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa flashattn_v0
python benchmarks/benchmark_decode.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa_naive flashattn_v0

# Generate plots
python benchmarks/plot_results.py

# Optional: build and numerically check the v1 prefill kernel
python -m csrc.compile --v1-only
pytest tests/test_correctness.py -vv

# Build and check the v2 GQA-native decode kernel
python -m csrc.compile --v2-only
pytest tests/test_v2_gqa_decode.py -vv
python benchmarks/benchmark_decode.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa_naive flashattn_v0 flashattn_v2 \
    --context-lens 512 1024 2048 4096 \
    --batch-sizes 1 4 \
    --check-correctness
```

## Latest Snapshot

The H100 roadmap plot is in `docs/figures/attention_roadmap_h100.png`.
V1 validates the bf16 WMMA prefill path, while V2 adds a GQA-native decode
kernel that reduces H100 PCIe decode p50 latency versus the v0 decode baseline
across batch 1/4 and context lengths 512-4096.

See `docs/PERFORMANCE.md` for additional notes and `docs/figures/` for
latency / TBT / speedup plots.

## Roadmap

- Mainline execution plan: `docs/PROJECT_ROADMAP.md`
- Canonical implementation plan: `docs/EXECUTION_PLAN.md`
- Documentation index: `docs/README.md`
