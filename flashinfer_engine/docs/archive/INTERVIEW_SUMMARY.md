# Interview Summary — LLM Inference Engine

A memorizable narrative for resume / interview discussion of this project.

---

## 30-Second Elevator Pitch

> "I built an LLM inference engine focused on the attention path — prefill,
> decode, and KV cache. I wrote custom CUDA FlashAttention kernels and built
> a benchmarking harness that measures them against PyTorch SDPA on real
> Llama-3 model shapes. The harness produces JSON results and plots, runs
> on H100, and is the foundation for ongoing kernel optimization work
> (Tensor Cores, GQA-native, Paged KV Cache)."

---

## 2-Minute Project Description

**Problem.** LLM inference is bottlenecked by the attention operation. Prefill
is compute-bound (O(n²) over sequence length), decode is memory-bandwidth-bound
(streaming the KV cache for every generated token). Serving systems like vLLM
and SGLang spend most of their engineering effort on these two paths.

**What I built.**
1. **Custom CUDA FlashAttention kernels** for prefill (online softmax tiling)
   and decode (single-token attention against a cached KV history) — fp32 baseline.
2. **A benchmark harness** (`flashinfer_engine`) that wraps any attention backend
   behind a uniform interface, sweeps shapes (batch × seq × head_dim), records
   p50/p95/p99 latency, throughput, and VRAM, and writes structured JSON.
3. **Baselines and comparisons**: PyTorch SDPA (cuDNN FlashAttention-2 internally)
   as the production reference; my v0 kernel as the educational implementation.
4. **Plotting pipeline** that turns the JSON results into latency curves,
   TBT curves, and speedup-vs-seq-len plots.

**Hardware.** Validated on H100 80GB. Model shapes: Llama-3.2-1B and Llama-3-8B
(GQA: 32 Q heads, 8 KV heads, head_dim=64-128).

**Engineering quality.** Project is structured as a pip-installable package
(`pyproject.toml`), backends register through a small dispatch layer, results
have a stable JSON schema with git commit + GPU name + timestamp, and the
plot script is deterministic and runs on cached results without a GPU.

---

## Key Technical Decisions

### 1. Two separate benchmarks, not one
Prefill and decode have fundamentally different bottlenecks (compute vs.
bandwidth), different shapes (full sequence vs. single token), and different
reporting metrics (TTFT vs. TBT). They share infrastructure but live in
separate scripts.

### 2. Backend abstraction over SDPA / kernels
A backend just exposes `prefill()` or `decode()` with `(batch, heads, seq, head_dim)`
inputs. Adding a new kernel version is one file plus a registry entry — no
benchmark code changes. Future v1/v2/v3 plug in here.

### 3. JSON-first results with stable schema
Every run writes one JSON per (batch, seq, backend) cell, with full identity
(git commit, GPU model, dtype, shape). This decouples measurement from
visualization — re-plot historical data anytime, diff results across commits,
detect regressions automatically.

### 4. Shape and naming aligned to industry models
Configs are real Llama-3.2-1B / Llama-3-8B shapes (GQA, head_dim=64/128,
hidden=2048/4096). When the kernel doesn't support GQA, the wrapper does
`repeat_interleave` and the cost is recorded — making the GQA gap visible
and quantified instead of hand-waved.

---

## Concrete Results (H100, fp32)

### Prefill (Llama-3.2-1B, batch=4)

| seq_len | SDPA p50 | flashattn_v0 p50 | gap |
|---------|----------|------------------|-----|
| 512     | 0.21 ms  | 0.95 ms          | 4.3x slower |
| 4096    | 8.30 ms  | 44.25 ms         | 5.3x slower |

**Scaling**: SDPA 8.30 / 0.21 ≈ 40x for 8x seq_len; v0 44.25 / 0.95 ≈ 47x.
Both follow O(n²) as expected; v0's coefficient is ~5x higher.

### Decode TBT (Llama-3.2-1B, batch=4)

| context_len | SDPA p50 | flashattn_v0 p50 | gap |
|-------------|----------|------------------|-----|
| 512         | 0.08 ms  | 0.18 ms          | 2.3x slower |
| 4096        | 0.47 ms  | 1.21 ms          | 2.6x slower |

**Decode gap is smaller than prefill gap** because decode is memory-bandwidth
bound, where SDPA's compute optimizations (Tensor Cores) help less.

### Memory
v0 uses ~2-3x more VRAM because it's MHA-only — a Llama-3 GQA model has 8 KV
heads but v0 needs them expanded to 32, 4x KV cache cost.

---

## Honest Assessment of the Gap

I can name the specific reasons v0 is 4-5x behind SDPA:

| Factor | SDPA (cuDNN) | v0 (mine) | Impact |
|--------|--------------|-----------|--------|
| Precision | fp32 (this run); auto-bf16 for prod | fp32 only | ~2x throughput in bf16 |
| Tensor Cores | wmma / wgmma | none | ~2-4x on compute-bound |
| GQA-native | yes | no — expand KV 4x | 4x VRAM, extra mem traffic |
| Tile schedule | per-arch tuned | fixed | ~10-20% on edge shapes |
| Warp specialization | yes | no | 10-30% on H100 |

The gap is **not** a defect — it's the gap between an educational
implementation and three years of NVIDIA + Tri Dao optimization.
The project's value is the **benchmark harness that quantifies the gap
component by component** and the roadmap to close it.

---

## What's Next (Roadmap)

| Milestone | Goal | Expected impact |
|-----------|------|-----------------|
| **I1** | bf16 + Tensor Cores (wmma) | Close prefill gap to ~2x |
| **I2** | GQA-native kernel | Eliminate 4x VRAM overhead |
| **I3** | Paged KV Cache | Enable long-context (32k–128k) and continuous batching |

I1 is the first real optimization milestone. The plan is in `docs/OPTIMIZATION_PLAN.md`.

---

## Likely Interview Questions & Answers

**Q: Why is your kernel slower than PyTorch?**
A: PyTorch SDPA dispatches to cuDNN FlashAttention-2, which uses Tensor Cores
(wmma), bf16, and per-architecture tile tuning. My v0 is fp32 with no Tensor
Core usage — it's the educational baseline. The point of building it isn't to
beat cuDNN, it's to (a) understand the algorithm at a tile level, and (b) have
a controlled implementation I can extend with paged KV, custom attention
patterns, or speculative decoding research.

**Q: Why measure prefill and decode separately?**
A: Different bottlenecks. Prefill is compute-bound — work scales O(n²), so
TTFT is dominated by GEMMs. Decode is memory-bandwidth-bound — every step
streams the entire KV cache for one new token, so TBT is dominated by HBM
reads. Optimizations for one don't transfer to the other; e.g., Tensor Cores
help prefill a lot but barely move decode.

**Q: What's the one most-impactful optimization to do next?**
A: bf16 + Tensor Cores via wmma. It's a direct 2x on memory, 4x on FMA
throughput on H100. Before that, paged KV or fancier tiling would be premature.

**Q: How do you avoid measurement noise?**
A: 5 warmup iterations, 20 timed runs, `torch.cuda.synchronize()` before and
after each. Report p50/p95/p99 — p99 within 5% of p50 across all runs is the
sanity check that the timing is stable.

**Q: Why Llama-3 specifically?**
A: It's the most common shape used by serving systems (vLLM, SGLang,
TensorRT-LLM benchmark against it). It exercises GQA (32 Q heads, 8 KV heads),
RoPE, and SwiGLU — the actual modern decoder block, not GPT-2's older shape.
My harness shapes match Llama-3.2-1B and Llama-3-8B exactly.

**Q: How would you scale this to multi-GPU?**
A: Tensor parallelism on the heads dimension is the standard answer — split
num_heads across GPUs, all-reduce after attention output projection. The
kernel signature stays the same; the engine layer manages the split.

**Q: How do you know the kernel is correct, not just fast?**
A: Two layers. (1) Numerical: `torch.allclose` against SDPA on small shapes
(planned in tests/). (2) End-to-end: I plugged it into GPT-2 inference and
compared generated text against the HuggingFace reference — same coherence
level means attention is computing correct values.

---
