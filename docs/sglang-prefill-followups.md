# SGLang Prefill Follow-Ups

Deferred ideas from the TP-only prefill investigation.

## Explicit MoE output ownership

In the current TP-only standard dispatcher path, the MoE backend allocates the
output activation buffer itself and returns it as the final `hidden_states`
tensor. In the common no-DP-attention case, `StandardDispatcher.combine(...)`
is effectively an identity path, so this allocation is already the final MoE
output buffer rather than an extra staging allocation.

Deferred idea:

- Move output ownership up to the layer/dispatcher boundary and pass
  `output=` into the backend explicitly.
- Once ownership is explicit, consider reusing a per-stream output buffer for
  repeated prefill shapes instead of allocating inside the backend each call.

Important constraint:

- Do not assume the TP-only MoE output can leave `use_symmetric_memory(...)`.
  Even though this path does not need all-to-all buffers, the output still
  feeds TP all-reduce, so NCCL/symmetric-memory-backed allocation may still be
  the right contract.

Expected value:

- Mostly allocator cleanup and reduced variance.
- Potential steady-state throughput improvement is probably small; treat this
  as a cleanup/follow-up item, not the main prefill optimization path.
