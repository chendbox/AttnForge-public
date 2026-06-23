# Execution Plan

This is the plan we will follow for the project.

The scope is intentionally limited to three stages:

1. `v1`: prefill optimization
2. `v2`: KV cache and decode optimization
3. `runtime`: lightweight inference integration

## Stage 1: `v1`

### Goal

Ship a benchmarked `flashattn_v1` prefill backend based on bf16 + Tensor Core
ideas.

### Tasks

1. Make sure `flashattn_v1` is discoverable by the backend registry
2. Add or adjust correctness tests for `v1`
3. Verify benchmark scripts can run `flashattn_v1`
4. Bring up the `v1` prefill kernel path
5. Compare `sdpa`, `flashattn_v0`, and `flashattn_v1`
6. Record the result in docs

### Exit Criteria

- `flashattn_v1` runs end to end
- correctness is validated
- benchmark evidence exists

## Stage 2: `v2`

### Goal

Upgrade the decode path to use a GQA-native KV cache.

### Tasks

1. Redefine KV cache layout around `num_kv_heads`
2. Update decode backend interfaces if needed
3. Update correctness tests for the new cache contract
4. Implement or adapt the decode path to use the new layout
5. Measure decode and memory behavior
6. Record the result in docs

### Exit Criteria

- decode works with GQA-native KV layout
- tests cover the new cache semantics
- benchmark or measurement evidence exists

## Stage 3: `runtime`

### Goal

Wrap the optimized paths in a lightweight inference runtime.

### Tasks

1. Define a small runtime entry path for prefill + decode
2. Centralize KV cache ownership and lifecycle
3. Connect runtime flow to the optimized backends
4. Add a small demo or integration path for MLIS use
5. Record the result in docs

### Exit Criteria

- runtime can drive prefill and decode
- KV cache is managed at the runtime layer
- a demo or integration path exists

## Working Rules

For every stage:

- make one clear change at a time
- keep tests passing
- do not expand scope before the current stage is working
- update the docs when the stage lands

## What We Do Next

We start with `v1`.

Immediate next tasks:

1. inspect backend registration for `flashattn_v1`
2. inspect benchmark entry points for `flashattn_v1`
3. inspect current tests and add the minimum needed coverage for `v1`
