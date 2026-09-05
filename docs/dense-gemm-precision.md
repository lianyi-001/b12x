# Dense GEMM activation precision

Status: implemented on SM120/SM121. `gemm.blockscaled` accepts BF16 activations with
NVFP4 or MXFP8 weights and selects an activation precision for dense linear
projections. MoE routing and expert GEMMs have separate implementations.

`mode="a16"` uses the BF16 warp-MMA specialization of
`b12x/_lib/dense_gemm.py::DenseGemmKernel`. The specialization retains the dense
engine's TMA producer, shared-memory pipeline, tile scheduler, accumulator
epilogue, compiler, and launch wrapper. It loads compressed weights and their
scales into shared memory, converts weight pairs directly into BF16 MMA
registers, and accumulates in FP32. Split-K writes FP32 partials and reduces
them into BF16 output. Activations remain BF16 throughout this route.

The implementation follows the inline weight conversion and narrow-M warp-MMA
patterns in `b12x/moe/_shared/kernels/w4a16/kernel.py` and the native FP8
conversion helpers in `b12x/_lib/intrinsics.py`. The MoE engine and its prepared
scale layout are references, not dependencies of the dense launch path. Triton
is used only for supporting activation quantization and packing in
`b12x/gemm/blockscaled/_quantize.py`.

## Shared weight contract

| Recipe | Stored values | Block scales | Reconstructed weight |
| --- | --- | --- | --- |
| NVFP4 | `uint8[N,K/2]`, low nibble first | E4M3, one per 16 K values | `E2M1 * block_scale * global_scale` |
| MXFP8 | `float8_e4m3fn[N,K]` | UE8M0, one per 32 K values | `E4M3 * block_scale` |

Both activation precision routes accept the same F8_128x4-swizzled weight
scale storage. `w4a16`/`w8a16` accept its flat physical storage or native six
dimensional MMA view. `pack_weight` also supports the established compact
MXFP8 scale input, which it swizzles during weight preparation.

NVFP4 `pack_weight` borrows the packed values and scale tensors without
rewriting them. `global_scale_kind="reciprocal"` interprets the supplied weight
global scale as a quantizer multiplier and divides by it in the epilogue.
Neither mode creates a second weight-scale tensor. Global scales must be
finite and positive; reconstructed weights must fit BF16.

A16 requires contiguous, 16-byte-aligned CUDA tensors, N divisible by 8,
stored K divisible by 32, and input K divisible by 8. Its native packed BF16
conversions require PTX 9.2 (CUDA 13.3). Quantized activation execution requires
stored K divisible by 128. MXFP8's established functional path remains available
for its other supported layouts and devices.

## One-shot calls and graph capture

```python
import torch
from b12x.gemm import blockscaled

# All tensors are already on the same SM120/SM121 device.
weight = blockscaled.pack_weight(
    packed_w, swizzled_weight_scales, recipe="nvfp4",
    global_scale=weight_global_scale, global_scale_kind="multiplier",
)
scratch = torch.empty(
    blockscaled.workspace_size(weight, max_tokens),
    dtype=torch.uint8, device=packed_w.device,
)
blockscaled.prewarm(
    weight, token_counts, workspace=scratch,
    activation_global_scale=activation_quantizer_multiplier,
)
blockscaled.mm(
    x, weight, out=y, workspace=scratch, mode="auto",
    activation_global_scale=activation_quantizer_multiplier,
)

# Standalone W4A16 consumes only weight scales.
blockscaled.w4a16(
    x, packed_w, swizzled_weight_scales, weight_global_scale, out=y,
)
```

Prewarm every precision/configuration route needed by the captured token
counts. The workspace is caller-owned and reusable across sequential calls;
concurrent calls require disjoint output and workspace buffers. No public
plan/bind/run API is introduced. A CUDA graph retains its captured precision
route. Kernel compilation and policy resolution use static geometry and device
identity; live M drives launch grids and masks.

## Measured dispatch

The registered offline component is `gemm.blockscaled_precision`:

```bash
.venv/bin/python scripts/generate_gpu_profile.py \
  --components gemm.blockscaled_precision --warmup 3 \
  --work-dir /tmp/blockscaled-profile-work \
  --output /tmp/blockscaled-profile.json
```

The corpus covers `(N,K)` equal to `(4096,5376)`, `(16384,1024)`,
`(17408,5120)`, and `(5120,17408)`, for both recipes at M=1 through 16 and
24, 32, 64, 128, 256, 512, 1024, and 2048. Each case races the quantized path
against sixteen A16 configurations. Small-M MXFP8 also includes the existing
fused activation-quantization GEMM as a baseline.

Timings include per-block activation scale computation, quantization, packing,
and GEMM. NVFP4's activation global scale is supplied as an input; computing
that scalar is outside the timed graph. Weight preparation is outside all
timings. Separate GEMM-only benchmark records are diagnostic and do not decide
precision promotion.

Candidates must pass independent numerical oracles and poisoned-buffer graph
replay before timing. A16 must have median latency no greater than every
quantized baseline in both an initial race and a separate confirmation pass.
The promotion threshold is 0%; equal latency prefers A16 because it preserves
BF16 activations. Bootstrapped 95% ratio intervals can be reconstructed from
the retained samples; they do not impose a statistically significant speedup
requirement. Paired replay order is balanced, L2 is flushed by default, and
raw samples, clock snapshots,
allocation checks, and source/toolchain identity are retained in checkpoints.
The Max-Q diagnostic clock contract allows only throttle masks 0x0/0x4, P1,
stable memory clocks, and a maximum 30 MHz SM-clock difference.

Runtime queries contain only recipe, input features, and output features.
An autotuned config stores exact measured M routes; it does not interpolate
an M threshold. M values omitted from that config retain quantized activation
execution. Measurements establish a projection's end-to-end crossover; they
do not isolate tensor-core instruction latency as its cause.

For an uncovered device or model geometry, the registered component heuristic
promotes NVFP4 to A16 at M=1 through 8 when K is at least 4096 and divisible by
128, N is divisible by 8, and `ceil(N/128) * 4 <= SM_count <= ceil(N/128) * 6`.
It chooses a 128-column, K=64 tile with four K slices. The output grid then
occupies between two thirds and one SM wave. Other NVFP4 cases and MXFP8 retain
quantized activations. This is a heuristic prediction, not a measured guarantee
for an uncovered geometry or device.

Autotuned entries take precedence over the heuristic, including an entry that
selects quantized activation execution for every M. `B12X_POLICY_MODE` supports
the existing `heuristic-only` and `preplanned-only` qualification modes. Explicit
`mode="a16"` or `mode="quantized"` bypasses promotion policy. Already-quantized
activation inputs retain their supplied precision. A16 promotion requires an
eligible BF16 input layout; MXFP8's functional API retains its established
handling of noncontiguous input.

## Qualified Max-Q coverage

The embedded profile `nvidia.rtx.pro.6000.blackwell.max-q` matches the NVIDIA
RTX PRO 6000 Blackwell Max-Q Workstation Edition with 188 SMs. Its precision
table is qualified by 192 GPU cases using a 5% selection margin. Regeneration
with the 0% rule is not qualified, so this embedded table has narrower promotion
coverage than the parity selector permits. All 3,264 case/candidate combinations
passed correctness, and the table contains 36 NVFP4 and 10 MXFP8 promotion routes:

| Weight recipe | N | K | M values selecting A16 |
| --- | ---: | ---: | --- |
| NVFP4 | 4096 | 5376 | 1–15 |
| NVFP4 | 16384 | 1024 | 15, 16 |
| NVFP4 | 17408 | 5120 | 1, 3, 7 |
| NVFP4 | 5120 | 17408 | 1–16 |
| MXFP8 | 4096 | 5376 | 14 |
| MXFP8 | 16384 | 1024 | 13 |
| MXFP8 | 17408 | 5120 | None |
| MXFP8 | 5120 | 17408 | 9–16 |

All measured M values from 24 through 2048 retain quantized activations.
Representative independent-confirmation medians are:

| Recipe | N | K | M | A16, µs | Quantized, µs | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVFP4 | 4096 | 5376 | 1 | 16.992 | 20.384 | 16.6% |
| NVFP4 | 4096 | 5376 | 8 | 15.040 | 18.336 | 18.0% |
| NVFP4 | 5120 | 17408 | 8 | 36.768 | 40.960 | 10.2% |
| MXFP8 | 5120 | 17408 | 16 | 57.248 | 62.048 | 7.7% |

These are cold-L2 graph timings with supplied activation global scales and a
325 W power limit. The toolchain was PyTorch 2.12.0+cu132, CUTLASS DSL 4.6.0,
Triton 3.7.0, and PTXAS 13.3.73. The physical GPU UUID was
`GPU-ac6fcbb2-ae5f-231d-cc3e-e843c305baff`. Runtime source was held fixed during
the qualifying run; the worktree was based on revision
`f2e8cd9214666645c5ea4994a9e5c479c587b2e8`.

Local evidence is `/tmp/b12x-precision-autotune.json`, its source manifest
`/tmp/b12x-precision-autotune-source.json`, and the raw checkpoints in
`/tmp/b12x-precision-autotune-work/`. The artifact SHA256 is
`e107b92b4e3977ee5dadfe00eca8d3c2863926dc87eb89ccb20fd391e6177e9f`;
the source-manifest SHA256 is
`19bf83e01aff2663aafe79f39d2ce5c06f6a2c81ab440653ce979c7d6f83143f`.
The qualifying command used `--warmup 25`. Thirteen checkpoint resumes were
needed to replace clock-rejected cases; their rejected samples are retained in
`/tmp/b12x-precision-autotune-rejected/`. A completed resume reused all 192
checkpoints and emitted a byte-identical profile.

Qualification covers the dense one-shot API. FlashInfer comparison checks were
skipped because FlashInfer was unavailable. The Max-Q measurements do not
qualify other RTX PRO 6000 product variants or SM121; those devices retain their
own profile coverage and heuristic behavior. The precision generator supports
SM120 and SM121.

## SM121 qualification

Status: correctness qualified on NVIDIA GB10, 48 SMs,
`GPU-87533355-db2d-9b70-eeab-5a9159ee4bc1`. The BF16 specialization selects its
architecture identity from the device and uses the same inline packed
conversions as SM120. With `CUTE_DSL_ARCH=sm_121a`, PyTorch 2.12.0+cu130,
CUTLASS DSL 4.6.2, and PTXAS 13.3.73, 58 targeted checks passed: exhaustive
FP4/FP8 value and scale conversion, both weight formats, split-K variants,
scale-tile tails, and graph replay under frozen kernel resolution.

Status: performance qualified for the four weight geometries and 24 row counts
listed above, for both recipes. All 3,264 candidate checks passed correctness;
all 192 cases passed timing qualification. The embedded profile contains 32
NVFP4 and 62 MXFP8 A16 promotion routes under the 0% threshold:

| Weight recipe | N | K | M values selecting A16 |
| --- | ---: | ---: | --- |
| NVFP4 | 4096 | 5376 | 1–12, 14–16 |
| NVFP4 | 16384 | 1024 | 1–14, 16 |
| NVFP4 | 17408 | 5120 | 1, 2 |
| NVFP4 | 5120 | 17408 | None |
| MXFP8 | 4096 | 5376 | 1–16 |
| MXFP8 | 16384 | 1024 | 1–14, 16, 24, 32 |
| MXFP8 | 17408 | 5120 | 1–3, 7–16 |
| MXFP8 | 5120 | 17408 | 1–16 |

Every promoted route passed the latency comparison in both independent timing
passes. All measured M values from 64 through 2048 retain quantized activations.
The uncovered-geometry heuristic also retains quantized activations on GB10.
Explicit `mode="a16"` is supported.

For N=4096, K=5376, representative independent-confirmation cold-L2 graph
medians in microseconds are:

| Recipe | M | Selected A16 | Fastest quantized baseline | A16 / quantized |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 | 1 | 59.456 | 61.408 | 0.9682 |
| NVFP4 | 8 | 59.424 | 61.440 | 0.9672 |
| MXFP8 | 1 | 104.448 | 106.496 | 0.9808 |
| MXFP8 | 8 | 107.808 | 108.544 | 0.9932 |

The ratio is A16 latency divided by quantized latency; lower is faster. Timings
include activation quantization under the API contract described above.
The device-specific timing gate requires P0, zero throttle mask, and at most
30 MHz SM-clock change between snapshots. NVML does not report the GB10 memory
clock; evidence records that limitation. SM120 Max-Q measurements retain their
separate timing gate.

The 25-warmup, 25-trial run completed without checkpoint retries in 11m06s.
The command was:

```bash
CUTE_DSL_ARCH=sm_121a \
CUDA_VISIBLE_DEVICES=GPU-87533355-db2d-9b70-eeab-5a9159ee4bc1 \
/home/luke/projects/vllm/.venv/bin/python scripts/generate_gpu_profile.py \
  --components gemm.blockscaled_precision --warmup 25 \
  --work-dir /tmp/b12x-sm121-parity-work \
  --output /tmp/b12x-sm121-parity.json
```

Evidence on `chroniton.local` is `/tmp/b12x-sm121-parity.json`, the source
manifest `/tmp/b12x-sm121-parity-source.json`, the selected-route audit
`/tmp/b12x-sm121-parity-audit.json`, and raw checkpoints under
`/tmp/b12x-sm121-parity-work/`. The artifact SHA256 is
`87f8ead04a89c27ee12993c7f80ca51462ea1fef073600c0b23ec936b1f3af49`;
the source-manifest SHA256 is
`ae7f817a3b0ba2d5b72c505b82daeba676eada8d6536ae0aebf0322af45ff3cc`.
The isolated source directory is `/home/luke/projects/b12x-precision-parity`,
based on revision `6698bee5f4793ac0139884439e1b4a0c621a39ba` with the
manifest-bound parity selection changes. The manifest records the source and
toolchain independently of installed package-version metadata.

With the embedded profile loaded, the A16 and precision-policy suites passed
96 checks; three SM120-specific checks were skipped. This includes BF16-reference
output on promoted decode routes and exact quantized output on retained
M=2048 routes under PREPLANNED_ONLY resolution, with poisoned-buffer graph
replay and frozen kernel resolution for both recipes.
