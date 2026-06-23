# Kernel Profile Report — flashattn_v0 vs PyTorch SDPA

**GPU:** NVIDIA H200 (SM 9.0, 132 SMs)  
**Tool:** Nsight Compute 2026.1.0  
**Shape:** Llama-3.2-1B · batch=1 · seq/ctx=1024 · heads=32 · head_dim=64 · fp32  
**Date:** 2026-05-16  

---

## 1. Kernel Identity

| Benchmark | Backend | Kernel name |
|-----------|---------|-------------|
| Prefill | flashattn_v0 | `flashattention_kernel<16, 16, 64>` |
| Prefill | SDPA | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` (cutlass MemEffAttn) |
| Decode  | flashattn_v0 | `flash_attention_decode_kernel<32>` |
| Decode  | SDPA | `fmha_cutlassF_f32_aligned_64x64_rf_sm80` |

> SDPA dispatches PyTorch's cutlass memory-efficient attention kernel — **not** the
> FlashAttention-3 WGMMA path, because this workload is fp32 and the shapes are small.

---

## 2. Summary Table

| Metric | v0 prefill | SDPA prefill | v0 decode | SDPA decode |
|--------|-----------|--------------|-----------|-------------|
| Kernel duration (µs) | 920 | ~90 | 187 | ~60 |
| Thread blocks launched | 2 048 | 512 | **32** | **32** |
| Full waves on H200 | 1.0 + partial | — | **0.06** | **0.06** |
| Achieved occupancy (%) | **78.4** | **15.5** | **12.5** | **6.3** |
| Theoretical occupancy (%) | 100 | 18.8 | 50 | 18.8 |
| FP32 pipeline utilization (%) | 8 | 3 | ~0 | ~0 |
| Dominant warp stall | shared bank conflict | register spill → L1TEX | L1TEX scoreboard 53% | fixed-latency dep. 33% |
| Shared bank conflicts | **29.2%** of store wavefronts | 19.2% loads / 12.8% stores | — | — |
| Uncoalesced shared accesses | 3% excess wavefronts | 15% excess wavefronts | — | — |

---

## 3. Prefill Analysis

### 3.1 flashattn_v0 prefill

**Grid:** `(64 blocks, 32 blocks, 1)` × `(256 threads, 1, 1)` = **2 048 thread blocks**

**Occupancy**
- Achieved: **78.4%** vs theoretical **100%**
- Theoretical capped at 62.5% due to **register pressure** (10 warps/scheduler vs hardware max 16)
- Gap between theoretical and achieved (~22%) from warp scheduling overhead and load imbalance

**Compute**
- FP32 pipeline at **8%** of H200 peak — far from compute-bound
- ncu reports "Compute and Memory are well-balanced" at current utilization level
- 606K fused + 338K non-fused FP32 instructions; converting non-fused → fused could yield ~18% gain

**Memory**
- **Shared memory bank conflicts: 29.2%** of store wavefronts affected (1.4-way average conflict)
  - Root cause: tile layout causes multiple threads to hit the same shared memory bank on writes
  - Fix: add padding (`__shared__ float tile[Br][Bc + 1]`) to stride accesses across banks
- Uncoalesced shared accesses: 3% excessive wavefronts (minor)

**Launch**
- **50% tail wave**: grid produces 1 full wave + partial wave of 992 blocks
  - Under uniform-duration assumption, partial wave ≈ 50% of total runtime is wasted
  - Fix: tune tile sizes so grid size is a multiple of SMs × blocks-per-SM

### 3.2 SDPA prefill (cutlass reference)

**Grid:** `(16 blocks, 32 blocks, 1)` × `(32 threads, 4, 1)` = **512 thread blocks**

**Occupancy**
- Achieved: **15.5%** vs theoretical **18.8%**
- Theoretical severely limited by both **register pressure** AND **shared memory** usage
  - 3 warps/scheduler vs hardware max 16 — heavy CUTLASS tile = large register file
- Despite very low occupancy, still ~10× faster than v0 at this shape

**Memory**
- **Register spilling to local memory: 81%** of all L1TEX sectors
  - Cutlass kernel is register-heavy; arrays that don't fit in registers spill to local memory
  - Local memory is cached in L1TEX (98.6% hit rate) so penalty is moderate
  - This is an accepted trade-off in production kernels: larger tiles → more register spill → but higher arithmetic intensity per global memory access

**Key insight:** SDPA achieves higher throughput not via occupancy but via
**higher arithmetic intensity** — each global memory load is reused across a larger tile before
being evicted, hiding latency even with few active warps.

---

## 4. Decode Analysis

### 4.1 flashattn_v0 decode

**Grid:** `(32 blocks, 1, 1)` × `(256 threads, 1, 1)` = **32 thread blocks**

**Critical finding: GPU severely underutilized**
- H200 has 132 SMs; 32 blocks = **0.06 full waves**
- 75.8% of SMs receive **no work at all**
- This is the primary reason decode is slower than it should be

**Occupancy**
- Achieved: **12.5%** vs theoretical **50%**
- Theoretical limited by registers + shared memory
- 75% gap between theoretical and achieved: scheduling overhead on nearly-empty GPU

**Warp stalls**
- Dominant stall: **L1TEX scoreboard 52.8%** of cycles
  - Warps stall waiting for global memory loads to return from cache/DRAM
  - With only 2 active warps/scheduler, no warps available to cover latency
  - Fix: more blocks (increase parallelism) or software prefetch

**Scheduler efficiency**
- Issues instruction only every **8.9 cycles** (ideal: every 1 cycle)
- Only 0.12 eligible warps/scheduler per cycle — almost always stalling

### 4.2 SDPA decode (cutlass reference)

**Grid:** `(1 block, 32 blocks, 1)` × `(32 threads, 4, 1)` = **32 thread blocks**

- Same 32-block grid → same underutilization problem
- Achieved occupancy: **6.3%** — even lower than v0 decode
- Dominant stall: **fixed-latency dependency 33%** (register pipeline latency, not memory)
  - This is actually a sign of better optimization: stalls are on cheap fixed-latency ops, not
    expensive DRAM reads — memory access is well-pipelined

> **Both decode kernels hit the same wall:** single-token decode with batch=1 produces only 32
> thread blocks, leaving most of the H200 idle. This is an algorithmic constraint, not an
> implementation bug.

---

## 5. Bottleneck Summary & Optimization Roadmap

### Prefill bottlenecks

| Priority | Issue | Est. Speedup | Fix |
|----------|-------|-------------|-----|
| 1 | Shared memory bank conflicts (29% stores) | ~25% | Pad shared tiles: `tile[Br][Bc+1]` |
| 2 | 50% tail wave (grid not multiple of wave size) | ~15% | Tune Br/Bc so grid = n × 132 |
| 3 | Non-fused FP32 instructions (35% non-fused) | ~18% | `-use_fast_math` or manual FMA |
| 4 | Register pressure limiting theoretical occupancy | ~37% | Reduce per-thread register usage (smaller tiles or `__launch_bounds__`) |

### Decode bottlenecks

| Priority | Issue | Est. Speedup | Fix |
|----------|-------|-------------|-----|
| 1 | Only 32 blocks for 132 SMs (0.06 waves) | **>2×** | Split work across heads or sequence chunks (Flash Decoding) |
| 2 | L1TEX scoreboard stall 53% | ~50% (after fix 1) | Prefetch K/V tiles while computing current tile |
| 3 | Single batch decode → trivially small grid | structural | Continuous batching: aggregate requests into larger batches |

### The root gap vs SDPA

SDPA's cutlass kernel achieves higher throughput through:
1. **Larger tiles** → higher arithmetic intensity per byte of global memory traffic
2. **Accepted register spilling** to L1TEX-cached local memory (a deliberate trade-off)
3. **Better instruction scheduling** — fixed-latency stalls instead of memory stalls in decode

v0 is a pedagogically correct FA-1 implementation. The path to close the gap:
```
v0 (FA-1, fp32, naive tiles)
  → v1: fp16/bf16 + Tensor Cores (wmma) → ~4× compute throughput
  → v2: FA-2 tiling (Q outer loop, better warp partition)
  → v3: Flash Decoding (split-K over sequence, reduce to partial softmax)
```

---

## 6. Raw Profile Files

Located at `benchmarks/profiles/`:
- `v0_prefill.ncu-rep` — view with `ncu-ui` for full roofline and source annotation
- `sdpa_prefill.ncu-rep`
- `v0_decode.ncu-rep`
- `sdpa_decode.ncu-rep`

Open interactively:
```bash
ncu-ui benchmarks/profiles/v0_prefill.ncu-rep
```
