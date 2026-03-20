# SGLang B12X Integration Notes

This note captures the current `sglang` backend boundaries and what they imply
for future b12x integration work.

The main question is not "can b12x swap in a different MoE kernel?" That part
already exists. The real question is which parts of the path `sglang` lets a
backend own:

- TP all-reduce / fused all-reduce + RMSNorm
- MoE dispatch format and transport
- expert compute
- output/combine ownership
- pre-dispatch work such as RMSNorm and routing

The answer is: some of this is already hookable, but full RMSNorm + dispatch
fusion is not available at the current MoE dispatcher boundary.

## Current boundaries

There are two separate backend interfaces in `sglang` MoE:

- `moe_a2a_backend`: owns dispatcher / transport / combine
- `moe_runner_backend`: owns expert compute

This split is selected in
`python/sglang/srt/layers/moe/fused_moe_triton/layer.py`
via `create_moe_dispatcher(...)`, and the runner side is mediated by
`python/sglang/srt/layers/moe/moe_runner/runner.py`.

For TP-only standard MoE:

- the model computes router logits and top-k before entering `self.experts(...)`
- the dispatcher then sees `hidden_states` plus `topk_output`
- the runner only sees `DispatchOutput`

This matters because the dispatcher boundary is already after:

- post-attention norm
- router/gate computation
- top-k expert selection

Representative model flow:

- `LayerCommunicator.prepare_mlp(...)` applies the post-attention norm before
  MoE / MLP entry in
  `python/sglang/srt/layers/communicator.py`
- sparse MoE blocks then compute `router_logits` and `topk_output` before
  calling `self.experts(hidden_states, topk_output)` in files such as:
  - `python/sglang/srt/models/qwen2_moe.py`
  - `python/sglang/srt/models/qwen3_moe.py`
  - `python/sglang/srt/models/deepseek_v2.py`
  - `python/sglang/srt/models/sarvam_moe.py`

So a backend that only plugs in at the current MoE dispatcher / runner level
does not get to fuse:

- post-attention RMSNorm
- gate / router matmul
- top-k selection

without changing higher-level model or communicator code.

## What existing backends fuse today

### 1. TP all-reduce + RMSNorm fusion

This is the main place where RMSNorm is already fused in `sglang`.

`LayerNorm.forward_with_allreduce_fusion(...)` in
`python/sglang/srt/layers/layernorm.py`
tries fused TP all-reduce + RMSNorm implementations before falling back to
manual all-reduce plus layernorm.

That path is triggered from `LayerCommunicator.prepare_mlp(...)` in
`python/sglang/srt/layers/communicator.py`.

The ownership point is not the MoE backend. It is the TP group /
custom-allreduce path:

- `tensor_model_parallel_fused_allreduce_rmsnorm(...)`
- `GroupCoordinator.fused_allreduce_rmsnorm(...)`
- `ca_comm.fused_allreduce_rmsnorm(...)`

So RMSNorm fusion today is a TP communication feature, not a MoE dispatch
feature.

Examples:

- AITER fused all-reduce + RMSNorm
- FlashInfer fused all-reduce + residual + RMSNorm
- PCIe oneshot fused all-reduce + RMSNorm through `ca_comm`

### 2. Attention-side fused norm + quant / prep

There are separate fused norm paths in attention preparation, for example in
DeepSeek attention:

- fused RMSNorm + FP8 quant
- fused RMSNorm + MXFP4 quant
- fused QK norm + RoPE

These live in attention code such as
`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
and are unrelated to the MoE backend boundary.

This is another useful precedent: `sglang` is willing to fuse norm with the
consumer, but only when the integration point is early enough.

### 3. FlashInfer dispatcher ownership

`FlashinferDispatcher` is the cleanest example of a backend owning the MoE
dispatch contract.

It controls:

- payload layout
- optional pre-dispatch FP4 quantization
- A2A transport
- workspace sizing
- output/combine buffer ownership

It can also provide a workspace-backed `moe_output` buffer so the MoE runner
writes directly into the combine payload.

This lives in
`python/sglang/srt/layers/moe/token_dispatcher/flashinfer.py`.

Important limit: even here, FlashInfer is not fusing post-attention RMSNorm
into MoE dispatch. The dispatcher still receives already-normalized
`hidden_states`.

### 4. DeepEP / Mooncake / Mori dispatcher ownership

These backends own transport and dispatch formats for EP MoE.

They introduce specialized `DispatchOutput` and `CombineInput` formats such as:

- `DEEPEP_NORMAL`
- `DEEPEP_LL`

and the runner side adapts to them through pre/post permute hooks in
`python/sglang/srt/layers/moe/moe_runner/base.py`.

This is evidence that `sglang` can support backend-specific dispatch contracts
cleanly, but again the integration point is after post-attention norm and after
top-k generation.

### 5. Ascend FuseEP: fused dispatch + expert compute

`NpuFuseEPDispatcher` is the strongest precedent for "dispatcher owns more than
transport."

In
`python/sglang/srt/layers/moe/token_dispatcher/fuseep.py`,
`dispatch(...)` directly calls `fused_deep_moe(...)` with:

- hidden states
- top-k ids / weights
- expert weights

and returns a dispatch output that is effectively already the expert result.

This shows that `sglang` will tolerate a dispatcher path that subsumes dispatch
and expert compute together.

Important limit: even this path still receives already-normalized
`hidden_states` and precomputed top-k. It does not own RMSNorm or router/top-k
generation.

## What b12x can own today

There are two separate viable hook points for b12x.

### A. TP transport / fused TP comm via `ca_comm`

`GroupCoordinator.all_reduce(...)` and
`GroupCoordinator.fused_allreduce_rmsnorm(...)` consult `self.ca_comm`.

That means b12x could own:

- TP all-reduce transport
- custom out-of-place all-reduce selection
- fused all-reduce + RMSNorm

by providing a custom all-reduce communicator in the `ca_comm` slot, similar to
the PCIe oneshot path in
`python/sglang/srt/distributed/device_communicators/custom_all_reduce.py`.

What this hook sees:

- just the TP reduction tensors
- optional residual / norm weight / eps for the fused norm path

What it does not see:

- MoE routing metadata
- expert ids
- top-k weights
- expert workspace

So this is a transport / TP-comm hook, not a full MoE integration hook.

### B. MoE dispatcher ownership

A b12x-owned dispatcher backend is also structurally viable.

This could be a TP-only backend even if it abuses the `moe_a2a_backend` slot
for a path that does not literally do all-to-all.

What that would let b12x own:

- dispatch output format
- workspace ownership
- per-stream reusable buffers
- output / combine buffer contract
- dispatch-time metadata packing
- specialized combine behavior
- fused dispatcher + runner pairings if needed

This is the right level for:

- explicit `output=` ownership
- b12x-owned workspace pools
- dispatch metadata tailored to b12x kernels
- future TP-only transport shortcuts near the MoE path

This is not enough for true RMSNorm + dispatch fusion because the dispatcher
still receives post-norm activations.

## What b12x cannot fully own at the current MoE boundary

At the current boundary, b12x cannot completely fuse:

- post-attention RMSNorm -> MoE dispatch
- gate matmul -> top-k -> dispatch

because those steps happen before `self.experts(...)`.

A b12x dispatcher can own the contract after those steps, but not those steps
themselves.

So if the real target is:

- RMSNorm
- maybe router/gate
- maybe top-k
- dispatch
- expert compute
- combine / output

then the current dispatcher boundary is too late.

## What would need to change for true RMSNorm + dispatch fusion

There are two plausible directions.

### Option 1: earlier MoE block entrypoint

Change sparse MoE blocks so the backend receives a richer input than
`(hidden_states, topk_output)`.

For example, a future backend-oriented entrypoint could receive:

- post-attention residual state
- layernorm weights / eps
- gate weights
- expert weights
- backend workspace / output buffers

This would let a backend own:

- post-attention RMSNorm
- gate
- top-k
- dispatch
- expert compute

This is the cleanest route to full integration, but it is a larger API change.

### Option 2: extend `prepare_mlp(...)`

Push more backend ownership into `LayerCommunicator.prepare_mlp(...)` or the
layer communicator contract.

This would let a backend fuse TP all-reduce + RMSNorm and possibly hand off a
backend-specific MLP/MoE payload directly to the sparse MoE block.

This is closer to current TP fused-allreduce integration, but still requires
coordination above the dispatcher level.

## Practical implementation order

If the goal is future integration without overcommitting to a large rewrite,
the sensible order is:

1. TP transport path:
   - If useful, add a b12x-owned `ca_comm` path for TP all-reduce and maybe
     fused all-reduce + RMSNorm.
   - Treat this as orthogonal to MoE.

2. TP-only b12x dispatcher backend:
   - Add a b12x dispatcher / transport contract under the MoE dispatcher slot.
   - Use it for workspace ownership, output ownership, and custom dispatch
     metadata.
   - This is the right step if the immediate goal is tighter MoE integration.

3. Earlier boundary for norm / routing fusion:
   - Only after the dispatcher path proves useful, decide whether to move the
     backend boundary earlier so b12x can own RMSNorm and possibly gate/top-k.

## Bottom line

`sglang` already supports backend ownership at multiple layers, but they are
different layers:

- TP fused norm lives in the TP communication / `ca_comm` path.
- MoE dispatch ownership lives in `moe_a2a_backend`.
- expert compute ownership lives in `moe_runner_backend`.

So:

- yes, b12x can plausibly own TP transport
- yes, b12x can plausibly own MoE dispatch
- no, current MoE dispatcher ownership alone is not enough for true
  RMSNorm + dispatch fusion

For full integration, b12x would eventually need both:

- a transport / dispatcher story
- an earlier model-side handoff than the current `(hidden_states, topk_output)`
  MoE boundary
