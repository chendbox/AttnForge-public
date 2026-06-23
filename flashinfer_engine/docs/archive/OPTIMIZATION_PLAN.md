# Optimization Plan — Closing the Gap to SDPA

This is the experiment-driven optimization roadmap for `flashattn_v0` →
`flashattn_v1/v2/v3`. It assumes you've read `INTERVIEW_SUMMARY.md` and the
latest numbers in `PERFORMANCE.md`.

---

## Where We Are Now

| Path | v0 vs SDPA | Driver |
|------|-----------|--------|
| Prefill | 4.0–5.3x slower | Compute bound, no Tensor Cores, fp32 |
| Decode  | 1.5–2.6x slower | Memory bound, KV expanded 4x for GQA |
| VRAM    | 2–3x more       | GQA-expansion overhead |

The current v0 kernel is the educational baseline. Don't apologize for the
gap — but don't optimize blindly either. Every change should be motivated by
a **profile-driven hypothesis**, not "what if I try X."

---

## Guiding Principles

1. **Profile first, optimize second.** Use Nsight Compute to identify the
   actual bottleneck (memory throughput, SM occupancy, warp stalls) before
   touching code.
2. **One variable per experiment.** Change one thing, re-run benchmark, log
   the delta in the JSON results. The harness already records `git_commit`
   per run — use branches and commits to track each experiment.
3. **Validate correctness on every change.** Numerical regression beats any
   speedup. Add `tests/test_correctness.py` before starting v1.
4. **Match the workload.** Don't optimize for shapes that don't matter.
   Llama-3-8B with seq=2k–8k is the realistic target; toy shapes lie.

---

## Phase 0 — Profile the Current v0 Kernel (1–2 days)

**Goal:** Quantify why v0 is slow. Don't write any new code yet.

### Experiments

**E0.1 — Roofline analysis**

Use Nsight Compute on the existing v0 prefill kernel:

```bash
ncu --set full --target-processes all -o v0_prefill_profile \
  python benchmarks/benchmark_prefill.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends flashattn_v0 \
    --batch-sizes 4 --prompt-lens 2048 --num-warmup 1 --num-runs 1
```

Record from the report:
- **SM occupancy** (target: >50%; expect: ~25-40%)
- **Memory throughput** (target: >70% of peak; expect: ~20-40%)
- **FMA throughput** (target: >50% of peak; expect: ~5-10% without TC)
- **Warp stall reasons** (top 3, ordered by % of cycles)

Compare same workload on SDPA:

```bash
ncu --set full -o sdpa_prefill_profile \
  python benchmarks/benchmark_prefill.py \
    --model configs/models/llama3_2_1b.yaml \
    --backends sdpa --batch-sizes 4 --prompt-lens 2048 --num-warmup 1 --num-runs 1
```

**Expected finding:** SDPA's FMA throughput will be 5-10x higher because of
Tensor Core usage. v0's bottleneck will be FFMA throughput (no Tensor Cores)
and likely "long scoreboard" stalls (memory waits).

**E0.2 — Decompose kernel time by stage**

Add `cudaEventRecord` markers in the v0 prefill source to measure:
- Q/K/V load from HBM to shared memory
- QK^T compute
- Online softmax (max + exp + sum)
- (softmax * V) compute
- Output store

This tells you **which stage** dominates. If softmax is 5% and matmuls are 90%,
all your optimization energy goes into matmuls.

**E0.3 — Same kernel at fp16 vs fp32 (PyTorch SDPA)**

```bash
python benchmarks/benchmark_prefill.py \
  --model configs/models/llama3_2_1b.yaml \
  --backends sdpa --dtype fp32
python benchmarks/benchmark_prefill.py \
  --model configs/models/llama3_2_1b.yaml \
  --backends sdpa --dtype bf16
```

This measures **how much SDPA itself benefits from bf16**. That sets your
ceiling for v1 — you can't beat SDPA in bf16 by going to bf16, you can only
catch up.

### Phase 0 Deliverable

`docs/PROFILE_REPORT.md` with the Nsight numbers and a one-paragraph
conclusion: "v0 is dominated by X (Y% of cycles), the highest-impact fix
is Z."

---

## Phase 1 — bf16 Path with Tensor Cores (`flashattn_v1`)

**Hypothesis:** Switching from fp32 FFMA to bf16 wmma will reduce prefill
gap from 4-5x to 1.5-2x.

**Why this is first:** Memory traffic halved (bf16 = 2 bytes), Tensor Core
throughput is 4x FFMA on H100 SM90.

### Experiments

**E1.1 — Naive bf16 conversion (no Tensor Cores yet)**

Just change all dtypes from fp32 → bf16. No wmma. Run benchmark.
Expected: ~1.5-2x speedup from halved memory traffic alone.

Branch: `v1-bf16-naive`. Commit + benchmark + JSON.

**E1.2 — Add wmma for the QK^T matmul**

Replace fp32 FFMA loop in QK^T with `nvcuda::wmma::mma_sync` using bf16
input, fp32 accumulate. Tile size constrained by wmma fragments (16×16×16 on
Ampere, 16×16×8 on H100 with hopper-specific intrinsics).

Branch: `v1-wmma-qkt`. Re-run benchmark. Expected: another 2-3x on prefill.

**E1.3 — Add wmma for the (softmax * V) matmul**

Same treatment for the second matmul.

Branch: `v1-wmma-full`. Expected: prefill gap drops to <2x.

### Phase 1 Success Criteria

- Prefill p50 ≤ 2.5x SDPA (down from 4-5x)
- Decode p50 ≤ 2x SDPA (down from 2.6x)
- Numerical: max abs error vs fp32 SDPA < 1e-2 (bf16 noise floor)
- VRAM: roughly halved from v0 (since cache is now bf16)

### Phase 1 Risks

- bf16 accumulation drift in long sequences. Mitigation: use fp32
  accumulator inside wmma fragments.
- wmma tile shape constraints may force a tile re-tune. Profile after E1.2;
  if SM occupancy drops, adjust BLOCK_M / BLOCK_N.

---

## Phase 2 — GQA-Native KV Cache (`flashattn_v2`)

**Hypothesis:** Storing KV at native `num_kv_heads` (not expanded to
`num_q_heads`) reduces KV traffic by 4x on Llama-3 (32 Q / 8 KV).

**Why this is second:** Decode is memory-bound. Cutting KV traffic 4x has a
direct impact on TBT. Also fixes the 2-3x VRAM blow-up.

### Experiments

**E2.1 — Wrapper-level: stop expanding KV before kernel call**

This is just changing the harness — it'll fail at runtime because the kernel
still assumes equal heads. The point is to confirm what the failure mode is
before changing the kernel.

Expected: shape mismatch crash. Confirms the kernel is the constraint.

**E2.2 — Kernel-level: read KV at native head count, broadcast in attention**

Each Q head reads from the same KV head as its sibling Q heads in the same
GQA group. The implementation is one extra index calculation:

```
kv_head_idx = q_head_idx / gqa_groups
```

Plus the cache shape becomes `(batch, num_kv_heads, seq, head_dim)`.

Branch: `v2-gqa-native`. Re-run benchmark.

### Phase 2 Success Criteria

- VRAM matches SDPA (no 2-3x blow-up)
- Decode TBT improves by 1.5-2x at large context (memory traffic drop)
- Prefill roughly unchanged (still compute-bound)

---

## Phase 3 — Paged KV Cache (`flashattn_v3`)

**Hypothesis:** Block-managed KV cache eliminates fragmentation and supports
continuous batching, both required for serving (not just single-request).

**This is the largest change** — touches both the kernel signature and the
engine's request management. Tackle only after Phase 1 and 2 are stable.

### Pre-work

- Read the vLLM PagedAttention paper (Kwon et al., SOSP 2023) — understand
  block table, physical/logical mapping, copy-on-write for prefix sharing.
- Look at FlashInfer's paged kernel signature for reference.

### Experiments

**E3.1 — Static cache → block-paginated cache, single request**

Just split the existing flat cache into 16-token blocks with an indirection
table. Single request only — no batching changes yet. Verify correctness
against v2.

**E3.2 — Multiple requests with different lengths, no batching**

Each request has its own block list. Kernel reads its blocks via the table.
Should be no slower than v2 single-request (just one indirection per token).

**E3.3 — Continuous batching: pack tokens from N requests into one kernel call**

This is where paged cache pays off. Add a request scheduler in the engine
layer (`src/flashinfer_engine/scheduler.py`).

### Phase 3 Success Criteria

- Throughput (tokens/s) improves 3-5x at 16+ concurrent requests vs
  per-request decode
- p99 TBT under 2x p50 (no head-of-line blocking)
- Long-context (32k) support: VRAM scales linearly with active tokens, not
  with `max_seq_len × concurrent_requests`

---

## Cross-Cutting Tasks (Do Anytime)

- **`tests/test_correctness.py`**: pytest comparing each backend against SDPA
  with `torch.allclose(rtol=1e-3, atol=1e-2)`. Mark `@pytest.mark.gpu`.
  Block all future PRs that fail this.
- **CI**: GitHub Actions workflow that runs CPU-only tests on every push and
  GPU benchmarks on a `[run-bench]` PR label.
- **`benchmarks/regression_check.py`**: load latest JSON, compare to previous
  commit's JSON, flag >5% latency regression.

---

## Anti-Patterns to Avoid

❌ **"Let me try a different tile size and see what happens."**
   Without a profile-driven reason, this is random walking. Profile first.

❌ **"I'll port FlashAttention-2 from scratch."**
   Way too much scope. Pick one optimization at a time. v1 = bf16+wmma only.

❌ **Optimizing for batch=1, seq=512.**
   Real workloads are batch=4-32, seq=2k-8k. Pick benchmark targets that
   reflect that.

❌ **Removing the SDPA baseline once v1 is faster.**
   Always keep the SDPA reference in benchmarks — it grounds every claim
   and protects against silent regressions.

❌ **Trying to beat cuDNN.**
   The goal is to be within 1.5-2x of SDPA at v1, then differentiate via
   features cuDNN doesn't have (paged cache, custom attention masks, fused
   speculative decoding).

---

## Suggested First Three Commits

1. `git checkout -b phase-0-profile` — add `docs/PROFILE_REPORT.md` with
   Nsight numbers from E0.1.
2. `git checkout -b v1-bf16-naive` — drop fp32 → bf16, no wmma. Benchmark.
3. `git checkout -b v1-wmma-qkt` — wmma on the first matmul only. Benchmark.

After each: re-run `benchmark_prefill.py` + `benchmark_decode.py`, save the
new JSONs, regenerate the speedup plot. The git history + JSON archive
becomes the experiment log.
