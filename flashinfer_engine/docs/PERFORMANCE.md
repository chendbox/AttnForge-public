# Performance Snapshot

Hardware: NVIDIA H100 80GB HBM3
Model: Llama-3.2-1B (32 Q heads, 8 KV heads, head_dim=64, fp32)
Date: 2026-05-09

## Prefill — `flashattn_v0` vs SDPA baseline

| batch | seq | SDPA p50 | v0 p50 | v0 / SDPA |
|-------|-----|----------|--------|-----------|
| 1 | 512 | 0.09 ms | 0.36 ms | 4.0x slower |
| 1 | 1024 | 0.23 ms | 1.00 ms | 4.3x slower |
| 1 | 2048 | 0.69 ms | 3.26 ms | 4.7x slower |
| 1 | 4096 | 2.32 ms | 11.65 ms | 5.0x slower |
| 4 | 512 | 0.22 ms | 0.94 ms | 4.3x slower |
| 4 | 1024 | 0.65 ms | 3.16 ms | 4.9x slower |
| 4 | 2048 | 2.23 ms | 11.55 ms | 5.2x slower |
| 4 | 4096 | 8.29 ms | 44.25 ms | 5.3x slower |

## Decode TBT — `flashattn_v0` vs SDPA naive

| batch | ctx | SDPA p50 | v0 p50 | v0 / SDPA |
|-------|-----|----------|--------|-----------|
| 1 | 512 | 0.077 ms | 0.112 ms | 1.5x slower |
| 1 | 1024 | 0.130 ms | 0.193 ms | 1.5x slower |
| 1 | 2048 | 0.239 ms | 0.427 ms | 1.8x slower |
| 1 | 4096 | 0.480 ms | 0.812 ms | 1.7x slower |
| 4 | 512 | 0.081 ms | 0.184 ms | 2.3x slower |
| 4 | 1024 | 0.137 ms | 0.331 ms | 2.4x slower |
| 4 | 2048 | 0.246 ms | 0.630 ms | 2.6x slower |
| 4 | 4096 | 0.466 ms | 1.214 ms | 2.6x slower |

## Why v0 is slower

| Factor | SDPA | flashattn_v0 |
|--------|------|--------------|
| Backing implementation | cuDNN FlashAttention-2 | educational tiled FA-1 |
| Precision | fp32 (in this run) | fp32 only |
| Tensor Cores | yes (wmma) | no |
| GQA-native | yes | no — KV expanded 4x |
| Tile schedule | per-arch | fixed |
| Warp specialization | yes | no |

## Optimization roadmap

- **v1**: bf16 + Tensor Cores (wmma) — expected 2-3x reduction on prefill gap
- **v2**: GQA-native — expected 3-4x VRAM reduction on Llama-style models
- **v3**: Paged KV Cache — required for serving long contexts and continuous batching

## Measurement method

- 5 warmup iterations, 20 timed runs per (backend, batch, shape) cell
- `torch.cuda.synchronize()` before and after each run
- p50 / p95 / p99 reported; tables show p50
- All raw data in `benchmarks/results/*.json`
- Reproducible via `benchmarks/benchmark_prefill.py` and `benchmarks/benchmark_decode.py`
