# SM120 Paged Attention Roadmap

## Goal

Build a real serving-grade SM120 paged-attention path in `b12x` for the Qwen-style full-attention case:

- causal self-attention
- `page_size=64`
- `8q:1kv` GQA
- `d=256`
- `bf16` Q/K/V first
- CUDA-graph-friendly

The immediate objective is to fix long-context scaling. The current kernel is correct, but structurally under-parallelized for this serving path.

## Real Shape To Optimize

Observed in SGLang/FlashInfer for Qwen TP=4:

- `q_shape=(48, 8, 256)`
- `k_cache_shape=(num_slots, 1, 256)` in the flat FlashInfer pool view
- `v_cache_shape=(num_slots, 1, 256)`
- `tp_q_heads=8`
- `tp_k_heads=1`
- `q_per_kv=8`
- causal
- `bf16` Q
- `fp8` KV in production, but `bf16` KV is milestone 1

For `b12x`, the first serving contract is:

- `q: [total_q, q_heads, d]`
- `k_cache, v_cache: [num_pages, page_size, kv_heads, d]`
- `page_table: [batch, max_pages_per_request]`
- `cache_seqlens: [batch]`
- `cu_seqlens_q: [batch + 1]`

## Current Diagnosis

The current paged kernel loses badly to FlashInfer `fa2` once context grows because:

1. Packed GQA collapses control-plane head parallelism.
2. There is no split-KV / split-K support.
3. Decode shares the same prefill-style paged kernel shape.

That combination leaves too few CTAs, and each CTA serially walks all K/V pages.

Relevant files:

- [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py)
- [block_info.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/block_info.py)
- [tile_scheduler.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/tile_scheduler.py)
- [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py)
- [benchmark_paged_attention.py](/home/luke/projects/b12x-research/rs-9/benchmarks/benchmark_paged_attention.py)

## Local Donors Inside `b12x`

The MoE path already has mature patterns that should be reused where they fit:

- exact workspace and workspace-pool contract:
  - [tp_moe.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/tp_moe.py)
- graph-capture safety tests and caller-owned output discipline:
  - [test_tp_moe_workspace_api.py](/home/luke/projects/b12x-research/rs-9/tests/test_tp_moe_workspace_api.py)
- graph replay equivalence under changing inputs:
  - [test_moe_equivalence.py](/home/luke/projects/b12x-research/rs-9/tests/test_moe_equivalence.py)
- multi-layer workspace and graph helper patterns:
  - [benchmark_moe.py](/home/luke/projects/b12x-research/rs-9/benchmarks/benchmark_moe.py)

These are the preferred local examples for:

- exact-shape workspace allocation
- workspace pools for non-graph flows
- caller-owned output buffers for graph capture
- capture/replay correctness tests
- reusable benchmark harness structure

## Guardrails

These are intentional constraints for the next phase:

- Keep packed GQA as the intra-CTA representation.
- Do not derive scheduler semantics from packed tensor shape.
- Do not start by disabling packed GQA to recover head parallelism.
- Do not start with a new scheduler; the existing split plumbing is already close.
- Do not spend the first cycle on `Q_in_regs`, `num_stages=2`, or tile retuning. Those may help constants, but not the slope.
- Do not build graph support around masked-off work inside one capture. Use discrete split buckets.

## Phase 0: Freeze The Baseline

### TODO

- Preserve the current benchmark as the baseline comparator.
- Keep FlashInfer `fa2` comparison in [benchmark_paged_attention.py](/home/luke/projects/b12x-research/rs-9/benchmarks/benchmark_paged_attention.py).
- Keep graph-capture + `100x` replay timing as the standard method.
- Keep the Qwen-like matrix:
  - `bs=8`
  - `q in {1, 6}`
  - `k in {64, 512, 2048, 8192}`
  - `page_size=64`
  - `8q:1kv`
  - `d=256`

### Exit Criteria

- Benchmark stays runnable on `CUDA_VISIBLE_DEVICES=7`.
- We can compare every architectural change against the same graph-captured matrix.

## Phase 1: Clean Up The Control Plane

Objective: make runtime scheduling reason about logical work, not packed storage layout.

### TODO

- Introduce explicit logical scheduler quantities in [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py):
  - `num_kv_heads`
  - `qhead_per_kvhead`
  - `logical_q_rows = q_len * qhead_per_kvhead`
  - `num_m_blocks = ceil_div(logical_q_rows, tile_m)`
- Stop deriving scheduler `num_head` and related control-plane values from packed `mQ.shape`.
- Keep `pack_gqa_layout(...)` as a layout transform only.
- Audit every use of packed shapes in scheduling, tile selection, and block-range logic.

### File Touchpoints

- [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py)
- [pack_gqa.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/pack_gqa.py)
- [tile_scheduler.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/tile_scheduler.py)

### Exit Criteria

- Scheduler arguments are built from explicit logical values.
- Packed GQA no longer implicitly changes launch semantics except where intended.
- Existing correctness tests still pass.

## Phase 2: Enable Split-KV In The Existing Kernel

Objective: unlock N/page parallelism without rewriting the scheduler.

### TODO

- Add runtime `num_splits` to the paged public API in [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py).
- Plumb `num_splits` into `TileSchedulerArguments` in [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py).
- Set `is_split_kv = (num_splits > 1)`.
- Stop hardcoding `num_splits=1`.
- Pass split mode into `BlockInfo(...)` instead of hardcoding `False`.
- In both load and MMA paths, unpack and use `split_idx` from `work_tile.tile_idx`.
- Thread `split_idx` and `num_splits` into `get_n_block_min_max(...)`.

### File Touchpoints

- [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py)
- [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py)
- [block_info.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/block_info.py)

### Exit Criteria

- Kernel launches with `num_splits > 1`.
- Different splits cover different K/V page ranges.
- Split and non-split modes match numerically after reduction.

## Phase 3: Implement Split Range Partitioning In `BlockInfo`

Objective: keep causal semantics correct while partitioning page work across splits.

### TODO

- Implement real split-aware range selection in [block_info.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/block_info.py).
- Start from the global masked range, then carve out a split-local subrange.
- Use contiguous chunks in reversed page order so split 0 gets the newest pages first.
- Keep descending local `n_block` traversal within each split.

Recommended first partition:

```python
global_min, global_max = masked_range(...)
tiles = global_max - global_min
chunk = ceil_div(tiles, num_splits)

split_max = global_max - split_idx * chunk
split_min = max(global_min, split_max - chunk)
```

### Exit Criteria

- Causal masking still behaves correctly on right-aligned serving shapes.
- Splits cover the global valid range exactly once with no gaps or overlap bugs.
- Existing paged reference tests can be extended to split mode and pass.

## Phase 4: Add A Two-Pass Split Reducer

Objective: get a correct prototype quickly without disturbing the online-softmax kernel math.

### TODO

- Add scratch buffers for per-split partial output and per-split LSE.
- Pass 1:
  - each split computes attention over only its assigned page range
  - write `O_i` and `LSE_i` to scratch
- Pass 2:
  - reduce splits per row
  - first attempt reducer formula:

```python
LSE = logsumexp_i(LSE_i)
O = Σ_i exp(LSE_i - LSE) * O_i
```

- Verify whether current `mLSE` is finalized row log-sum-exp after `Softmax.finalize()`.
- If not, expose the right reduction state instead, such as `(m_i, l_i)` or unnormalized partial output.

### File Touchpoints

- [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py)
- [softmax.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/softmax.py)
- [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py)
- tests under [tests](/home/luke/projects/b12x-research/rs-9/tests)

### Exit Criteria

- Split and non-split outputs agree within current tolerances.
- Reducer works under graph capture.
- Benchmark shows the long-context slope improves materially.

## Phase 5: Make Split Selection Graph-Safe

Objective: choose split count without introducing graph-hostile control flow.

### TODO

- Add discrete split buckets, starting with `{1, 2, 4, 8}`.
- Capture separate graphs per split bucket.
- Use a simple first heuristic based on page count.
- Second heuristic, if needed:
  - choose the smallest power-of-two split count that gets total CTAs high enough
  - keep a minimum number of pages per split so reduction overhead does not dominate
- Reject or fall back for unsupported split choices rather than masking work inside one graph.

### File Touchpoints

- [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py)
- future SGLang backend shim under [b12x/sglang](/home/luke/projects/b12x-research/rs-9/b12x/sglang)

### Exit Criteria

- Capture and replay work for each split bucket.
- Runtime chooses a bucket deterministically from static inputs.
- No graph-time host control flow is required inside the captured region.

## Phase 6: Add Focused Tests

Objective: keep the new architecture correct while it evolves.

### TODO

- Add split-KV correctness tests for paged causal attention.
- Add cases for:
  - decode `q=1`
  - extend `q=6`
  - `k in {64, 512, 2048, 8192}`
  - `num_splits in {1, 2, 4, 8}`
- Add graph-capture tests for the supported split buckets.
- Keep the existing FlashInfer comparison benchmark as the performance regression test.

### Exit Criteria

- Tests catch split-range bugs, reduction bugs, and graph incompatibilities.
- Benchmarks report both absolute timings and `fa2/b12x` ratio on the Qwen matrix.

## Phase 7: Only Then Tune Constants

Objective: improve constants after the architecture is fixed.

### TODO

- Evaluate `Q_in_regs=True` on the paged serving path.
- Evaluate `num_stages=2` for paged K/V.
- Re-check tile choices after split-KV is working.
- Re-run the full graph-captured benchmark matrix after every tuning change.

### Exit Criteria

- Improvements show up as better constants on top of the improved split-KV slope.
- No regression on correctness or graph capture.

## Phase 8: Decode-Specialized Kernel Family

Objective: optimize the `q=1` fast path after the split-KV architecture is in place.

### TODO

- Add a separate decode-oriented kernel family for tiny packed M.
- Do not try to force the current kernel into a micro-M shape; its structure is built around a larger M tile and current warp partitioning.
- Reuse the same paged-KV and split-KV serving contract.

### Exit Criteria

- Decode path outperforms the shared paged kernel on `q=1`.
- Extend path remains on the main kernel unless a separate path is justified.

## Phase 9: FP8 KV Cache

Objective: move from the milestone-1 `bf16` KV path to production-relevant mixed precision.

### TODO

- Add mixed `bf16` Q + `fp8` KV support.
- Define scale/descale handling in the public API.
- Validate against the real Qwen serving path after the `bf16` split-KV architecture is stable.

### Exit Criteria

- `fp8` KV path matches reference within acceptable tolerance.
- Graph capture remains intact.
- Performance is measured on the same Qwen matrix.

## Success Metrics

The kernel is on the right path when all of the following are true:

- Long-context latency no longer scales almost linearly with page count.
- `fa2/b12x` ratio materially improves at `k >= 512`.
- Graph capture works for the supported split buckets.
- Packed GQA remains enabled, but no longer silently dictates scheduler semantics.
- The public serving contract stays narrow and explicit.

## Immediate Next Tasks

- [ ] Refactor scheduler inputs in [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py) so logical dimensions are explicit and not inferred from packed layout.
- [ ] Add `num_splits` to the paged API in [attention.py](/home/luke/projects/b12x-research/rs-9/b12x/integration/attention.py).
- [ ] Plumb split mode into [forward.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/forward.py) and [block_info.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/block_info.py).
- [ ] Implement split-aware `n_block` partitioning in [block_info.py](/home/luke/projects/b12x-research/rs-9/b12x/attention/block_info.py).
- [ ] Add per-split scratch and a two-pass reducer.
- [ ] Add split-KV correctness tests and graph-capture tests.
- [ ] Re-run [benchmark_paged_attention.py](/home/luke/projects/b12x-research/rs-9/benchmarks/benchmark_paged_attention.py) against FlashInfer `fa2`.
