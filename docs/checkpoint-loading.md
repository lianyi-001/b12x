# Proposed checkpoint loader

Status: design, September 5, 2026. `b12x.loader` is not implemented yet.
Implementation is paused while vLLM switches from the custom InstantTensor
loader to fastsafetensors and removes the copy/buffer workarounds.

Add bounded checkpoint loading to b12x. The component owns checkpoint manifests,
I/O scheduling, buffer capacity, CUDA completion, and generic weight transforms.
Serving integrations supply model mappings, source/destination slices,
dependencies, required numerical semantics, and capacity limits. This follows
the existing rule that b12x owns planning and policy.

## Package boundary

Proposed layout:

```text
b12x/loader/
    __init__.py         lazy public exports
    api.py              manifest, plan, and session interface
    _contract.py        references, ranges, destinations, capabilities
    _manifest.py        indexed safetensors discovery and validation
    _planner.py         dependency scheduling, coalescing, budget admission
    _session.py         ownership, submission, retirement, error handling
    _native.py          build/cache and C ABI bindings
    _transport.cpp     batched positional I/O and CUDA transfers
    _transforms.py      bounded quantization/packing using existing b12x ops
b12x/integration/vllm/
    loader.py           public vLLM loader registration and adapters
```

The loader imports without vLLM. Importing `b12x` or `b12x.loader` must not build
native code, initialize CUDA, or import the kernel compiler. An explicit
session/preparation operation loads required dependencies before serving.

This is host-side startup infrastructure, with a session lifecycle rather than
an inference kernel's graph-replay lifecycle. Do not put disk I/O or blocking
waits in CUDA graph capture. GPU transforms use existing b12x component plans
and policy; any new planned GPU op must satisfy catalog/profile registration
and frozen-resolution requirements. Keep normal namespace/registry imports
lightweight and extend their existing tests.

## Ownership and memory

`WeightRef` keeps immutable source metadata, not a transient tensor. A planned
operation names its source ranges and final owned destinations. Its temporary
storage is reserved before scheduling. Arbitrary model callbacks never receive
a reusable staging view.

Host slots remain live through reads and transfers; GPU workspace remains live
through its last transform consumer. Requests hold strong references to storage
owners and file handles until completion. Retirement uses explicit CUDA events,
including all participating streams. Error/cancellation handling drains users
before unregistering or freeing memory and never publishes partial weights.

One session cap includes pinned host buffers, GPU staging, retained source
materializations, transform workspace, and alignment overhead. Target and draft
share that cap. On GB10, charge host and CUDA staging together because they
consume the same physical memory pool. Final model storage, metadata, CUDA
overhead, and external memory use are separately included in capacity checks.
Do not describe the staging cap as a bound on total system memory or page cache.

Start qualification with 256 MiB aggregate staging and 8 MiB read chunks; tune
only from measured loading results. Large tensors stream into final storage or
through bounded transform tiles. Tensor size must not trigger a full CPU tensor
fallback or silently enlarge the cap. Reserve progress workspace before
prefetching to prevent capacity deadlock.

## Native helper and integration

Use the native-helper pattern already present in `comm.roce`: packaged source,
a small C ABI, ctypes bindings, and a cached host-compiler build. Keep batches
inside native code to avoid Python calls per small tensor. Use liburing and
CUDA transfer/event APIs; no PyTorch C++ ABI or new pybind11 dependency is needed
for this interface. CUDA storage ownership remains explicit in Python session
objects until native completion.

Include source, ABI version, CPU architecture, compiler identity, flags, and
relevant dependency identity in the build-cache key. Build atomically and
validate the ABI on load. Build/probe failures must be clear, and preparation
must finish before inference warmup/frozen kernel resolution. Validate both
aarch64 and x86_64 builds and include native source in package data.

The first transport supports local buffered reads, asynchronous copies, and
bounded GPU transforms. A threaded positional-read implementation is a test
reference for the same range contract. Direct I/O and alternate transports
require measurements and must preserve the same ownership/capacity contract.

Register `--load-format b12x` through vLLM's public `register_model_loader`
interface using a dedicated general-plugin registration function. Keep plugin
registration cheap and idempotent in spawned workers; initialize transport only
on an actual load. The existing FP6 plugin remains independently selectable.
Model-specific Qwen/GLM descriptions may require vLLM changes; the plugin does
not eliminate that integration work. Integrations describe numerical recipes
and slices, while b12x selects chunking, batching, workspace, and execution order.

Reuse existing MXFP8/NVFP4 quantizers and packing implementations where their
output, rounding, scale domain, and layout match. Add bounded output interfaces
where necessary rather than duplicating quantization math in the integration.
The NVFP4 head needs a first pass over its existing global-scale domain before
chunk quantization. Shared target/draft source reads must preserve their
different final precision requirements.

## Qualification

Add behavior tests under `tests/loader/` and startup/transport benchmarks under
`benchmarks/loader/`. The existing `benchmarks/checkpoint_loader.py` is a small
indexed safetensors helper, not the streaming implementation.

Test ownership under repeated slot reuse, delayed and multi-stream consumers,
partial reads, cancellation, allocation failures, aliases, and paired scales
in either order. Include tensors larger than the cap, 64-bit offsets above
4 GiB, packed dtypes, TP slices, and checkpoint index overlays. Check actual
host/device high-water marks and memory release at session close.

Compare loaded bytes and transformed weights with trusted safetensors and the
existing quantizers before measuring speed. Qualify Qwen TP1 and GLM TP2 with
MTP off/on, cold/warm cache, and the real GB10 loading paths; include RTX for
discrete-memory validation. Record command, revision, physical GPU, memory
budget, correctness state, physical/selected bytes, raw timings, and comparison
direction. Do not switch the serving default until correctness, bounded memory,
and startup performance pass. The full model integration and acceptance plan
is in the companion vLLM checkout's `docs/design/streaming_weight_loading.md`.
