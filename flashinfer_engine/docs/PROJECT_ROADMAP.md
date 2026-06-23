# Project Roadmap

This project is no longer trying to cover every possible FlashAttention
milestone. The goal is a focused, resume-ready engineering story:

1. `v1`: speed up prefill with a bf16 + Tensor Core path
2. `v2`: improve decode with a GQA-native KV cache
3. `runtime`: integrate the optimized kernels into a lightweight inference
   runtime that can later connect to the MLIS project

This is the mainline plan we will follow.

## Current Baseline

The repository already has:

- `flashattn_v0` prefill: fp32, tiled FlashAttention-1-style baseline
- `flashattn_v0` decode: single-token decode with a static KV cache
- benchmark scripts, backend abstraction, and correctness tests

That means the next work is not "start from zero." The next work is to turn
the current kernel baseline into a small but credible inference project.

## Mainline

| Stage | Goal | Why it matters |
|-------|------|----------------|
| `v1` | bf16 + Tensor Core prefill path | closes the biggest compute gap vs SDPA |
| `v2` | GQA-native KV cache for decode | matches modern LLM inference better than MHA-only cache layout |
| `runtime` | lightweight inference integration | turns kernel work into a system-level project |

## Stage 1: `v1`

### Problem

The current `v0` prefill path is much slower than PyTorch SDPA because it is an
fp32 baseline and does not use the modern Tensor Core path.

### Scope

- bring up `flashattn_v1` as a real backend
- use bf16 input where appropriate
- use Tensor Core acceleration for the dominant compute path
- validate correctness against SDPA
- benchmark `sdpa` vs `flashattn_v0` vs `flashattn_v1`

### Deliverable

A benchmarked and testable `v1` prefill backend that clearly explains:

- what made `v0` slow
- what changed in `v1`
- how much prefill latency improved

### Resume value

"Upgraded a custom FlashAttention baseline to a bf16/Tensor Core prefill
backend and benchmarked it against PyTorch SDPA."

## Stage 2: `v2`

### Problem

The current decode path uses an MHA-oriented KV cache layout. That is not the
right abstraction for real GQA models such as Llama-family models, and it
wastes bandwidth and memory.

### Scope

- redesign the cache around `num_kv_heads`
- stop treating KV as if it had to match `num_q_heads`
- update decode-side backend contracts and tests
- benchmark decode behavior and memory impact

### Deliverable

A `v2` decode path with a GQA-native KV cache that clearly explains:

- why the old cache layout was inefficient
- how the new layout better matches production LLMs
- how decode/runtime memory behavior improved

### Resume value

"Designed a GQA-native KV cache layout for LLM decode, reducing bandwidth and
memory waste compared with an MHA-only baseline."

## Stage 3: `runtime`

### Problem

Kernel benchmarks alone are not enough for a strong project story. The kernels
should plug into a small inference runtime so the work can connect to a real
ML system, including the MLIS project.

### Scope

- build a lightweight runtime wrapper around the optimized backends
- support a simple request flow for prefill + decode
- manage KV cache lifecycle in one place
- add only minimal batching or scheduling if it helps the demo

This is intentionally not a full vLLM reimplementation.

### Deliverable

A lightweight runtime that can:

- run prefill and decode through the custom backends
- own KV cache management
- serve as an integration point for MLIS or a small serving demo

### Resume value

"Integrated custom CUDA attention kernels into a lightweight inference runtime
with KV cache management and model-serving hooks."

## Explicit Non-Goals

To keep the project focused, we are not treating these as mainline goals:

- full FlashAttention-3 implementation
- full vLLM reimplementation
- paged attention as a required milestone
- large-scale serving infrastructure

If time allows, those can become follow-up experiments, not blockers.

## Time Estimate

This is the working estimate for a resume-oriented version:

- `v1`: about 1 week
- `v2`: about 1 to 1.5 weeks
- `runtime`: about 1 week

Total expected effort: about 3 weeks.

## Execution Rule

We will move stage by stage.

1. Finish `v1`
2. Then move to `v2`
3. Then build the lightweight runtime

Each stage should end with:

- passing tests
- benchmark or demo evidence
- a short documentation update describing the result

## Source Of Truth

The canonical step-by-step execution plan lives in:

- `docs/EXECUTION_PLAN.md`

That file is what we will follow while implementing the project.
