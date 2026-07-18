# V3 PagedAttention MVP Design

## Goal

V3 extends the current GQA-native decode path from a static contiguous KV cache to a
PagedAttention-style KV cache. The goal is to make the project cover both custom
attention kernels and LLM serving-oriented KV-cache memory management.

This is an MVP, not a full vLLM runtime.

## Motivation

V2 decode uses a static GQA KV cache:

```text
k_cache/v_cache: (batch, num_kv_heads, max_seq_len, head_dim)
```

This is simple and fast to index, but every request reserves a contiguous
`max_seq_len` region. In serving workloads, requests have different prompt and
generation lengths, so static allocation can waste GPU memory and makes dynamic
batching harder.

PagedAttention stores KV cache in fixed-size blocks and uses a per-request block
table to map logical token positions to physical KV blocks.

## Scope

In scope:

- Paged GQA KV cache layout.
- Block table metadata.
- Single-token paged decode kernel.
- Correctness tests against a PyTorch SDPA reference.
- Lightweight memory benchmark comparing static and paged KV allocation.

Out of scope:

- Full vLLM runtime.
- Continuous batching scheduler.
- Prefix sharing.
- Preemption, swap, or eviction.
- FP8/INT4 quantization.
- FlashAttention-3.

## Proposed Layout

Static V2 layout:

```text
k_cache/v_cache: (batch, num_kv_heads, max_seq_len, head_dim)
```

Paged V3 layout:

```text
k_cache/v_cache: (num_blocks, num_kv_heads, block_size, head_dim)
block_table:     (batch, max_blocks_per_request)
context_lens:    (batch)
```

For a request `b` and logical token position `t`:

```text
logical_block = t / block_size
block_offset  = t % block_size
physical_block = block_table[b, logical_block]

k = k_cache[physical_block, kv_head, block_offset, dim]
v = v_cache[physical_block, kv_head, block_offset, dim]
```

For GQA head mapping:

```text
group_size = num_q_heads / num_kv_heads
kv_head = q_head / group_size
```

## Kernel API Sketch

The first CUDA extension can expose:

```text
paged_decode(
    q,
    k_cache,
    v_cache,
    block_table,
    context_lens,
    block_size,
) -> out
```

Expected shapes:

```text
q:           (batch, num_q_heads, 1, head_dim)
k_cache:     (num_blocks, num_kv_heads, block_size, head_dim)
v_cache:     (num_blocks, num_kv_heads, block_size, head_dim)
block_table: (batch, max_blocks_per_request)
context_lens:(batch)
out:         (batch, num_q_heads, 1, head_dim)
```

The MVP can assume:

- `q`, `k_cache`, and `v_cache` are fp32 initially.
- `head_dim` is 64.
- `block_size` is fixed per benchmark run, for example 16 or 32.
- Blocks are already allocated and populated before decode.

## Correctness Strategy

Tests should construct both a paged KV cache and an equivalent dense KV cache.

1. Generate random dense K/V tensors.
2. Scatter dense K/V into paged blocks using `block_table`.
3. Run V3 paged decode.
4. Run PyTorch SDPA on the dense reference.
5. Compare outputs with fp32 tolerances.

This verifies the block-table indexing, GQA head mapping, and online softmax path
without requiring a full runtime allocator.

## Benchmark Strategy

Latency benchmark:

- Compare V2 static decode vs V3 paged decode.
- Use batch sizes 1 and 4.
- Use context lengths 512, 1024, 2048, and 4096.
- Report p50, p95, and p99.

Memory benchmark:

- Compare allocated KV capacity for static vs paged layouts.
- Use mixed request lengths to show wasted capacity in static allocation.
- Report total allocated tokens, used tokens, and waste ratio.

Expected outcome:

- V3 may not beat V2 single-request latency because paged decode adds block-table
  lookup overhead.
- V3 should improve memory utilization for variable-length requests and provides
  the foundation for continuous batching.

## PR Plan

PR 1: Add paged KV metadata helpers.

- Define block-table utilities.
- Add dense-to-paged test helpers.
- Add memory accounting utilities.

PR 2: Add V3 paged decode CUDA extension.

- Add CUDA kernel and setup file.
- Add `--v3-only` compile support.

PR 3: Register V3 backend and correctness tests.

- Add Python backend wrapper.
- Register `flashattn_v3`.
- Add paged decode correctness tests.

PR 4: Add benchmarks and docs.

- Add paged decode latency benchmark.
- Add static-vs-paged KV memory benchmark.
- Add README summary and figure.

## Resume Framing

If completed, this milestone can be described as:

```text
Implemented a PagedAttention-style GQA KV-cache layout with block-table based
decode, profiling latency and memory tradeoffs for variable-length LLM inference
workloads on H100.
```
