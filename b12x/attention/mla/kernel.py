"""Correctness-first CuTeDSL sparse MLA kernel for the NSA packed-cache contract."""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import os
import warnings

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from b12x.attention import utils as attention_utils
from b12x.cute.fp4 import bfloat2_to_float2_scaled, fp8x4_e4m3_to_bfloat2x2
from b12x.cute.utils import current_cuda_stream

from .reference import _MLA_GROUP_SIZE, _MLA_NOPE_DIM, _MLA_PACKED_DIM, _MLA_ROPE_DIM
from .traits import SparseMLATraits, select_sparse_mla_traits


_MLA_HEAD_DIM = _MLA_NOPE_DIM + _MLA_ROPE_DIM
_MLA_SCALE_GROUPS = _MLA_NOPE_DIM // _MLA_GROUP_SIZE
_MLA_OUTPUT_DIM = _MLA_NOPE_DIM
_MLA_WARP_THREADS = 32
_MLA_OUTPUT_FRAGMENTS_PER_LANE = _MLA_OUTPUT_DIM // _MLA_WARP_THREADS
_EAGER_HOST_LAUNCHER_CACHE_SIZE = 32


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    if dtype == torch.uint8:
        return cutlass.Uint8
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
    *,
    assumed_align: int = 16,
) -> cutlass.cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    leading_dim = next((idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None)
    if leading_dim is not None and tensor.ndim >= 2:
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_meta_key(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...], tuple[int, ...], str, tuple[str, int | None]]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _launcher_cache_lookup(
    kernel: object,
    cache_key: tuple[object, ...],
):
    cache = getattr(kernel, "_eager_host_launchers", None)
    if cache is None:
        cache = OrderedDict()
        setattr(kernel, "_eager_host_launchers", cache)
        return cache, None
    compiled = cache.get(cache_key)
    if compiled is not None:
        cache.move_to_end(cache_key)
    return cache, compiled


def _run_cached_host_launcher(
    kernel: object,
    cache_key: tuple[object, ...],
    args: tuple[object, ...],
) -> None:
    cache, compiled = _launcher_cache_lookup(kernel, cache_key)
    if compiled is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Cache is disabled as user wants to compile only.",
                category=UserWarning,
            )
            compiled = kernel(*args, compile_only=True)
        cache[cache_key] = compiled
        if len(cache) > _EAGER_HOST_LAUNCHER_CACHE_SIZE:
            cache.popitem(last=False)
    exe_args, _ = compiled.generate_execution_args(*args)
    compiled.run_compiled_program(exe_args)


@cute.jit
def _warp_allreduce_sum(value: Float32) -> Float32:
    for shift in cutlass.range_constexpr(5):
        value = Float32(value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift))
    return value


@cute.jit
def _load_scaled_fp8x4(
    mQuant: cute.Tensor,
    mScales: cute.Tensor,
    token_idx: Int32,
    group_idx: cutlass.Constexpr[int],
    lane: Int32,
):
    base = Int32(group_idx * _MLA_GROUP_SIZE) + lane * Int32(4)
    packed = (
        Uint32(mQuant[token_idx, base + Int32(0)])
        | (Uint32(mQuant[token_idx, base + Int32(1)]) << Uint32(8))
        | (Uint32(mQuant[token_idx, base + Int32(2)]) << Uint32(16))
        | (Uint32(mQuant[token_idx, base + Int32(3)]) << Uint32(24))
    )
    scale = Float32(mScales[token_idx, Int32(group_idx)])
    bf2_01, bf2_23 = fp8x4_e4m3_to_bfloat2x2(packed)
    v0, v1 = bfloat2_to_float2_scaled(bf2_01, scale)
    v2, v3 = bfloat2_to_float2_scaled(bf2_23, scale)
    return v0, v1, v2, v3


class SparseMLAKernel:
    """Warp-per-query-head sparse MLA kernel for the exact GLM-5.1 contract."""

    @cute.jit
    def __call__(
        self,
        q_all: cute.Tensor,
        kv_nope_q: cute.Tensor,
        kv_scales: cute.Tensor,
        kv_rope: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_all,
            kv_nope_q,
            kv_scales,
            kv_rope,
            page_table_1,
            sm_scale,
            output,
        ).launch(
            grid=(q_all.shape[0], q_all.shape[1], 1),
            block=[_MLA_WARP_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_all: cute.Tensor,
        kv_nope_q: cute.Tensor,
        kv_scales: cute.Tensor,
        kv_rope: cute.Tensor,
        page_table_1: cute.Tensor,
        sm_scale: cute.Tensor,
        output: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        q_idx, head_idx, _ = cute.arch.block_idx()

        acc = cute.make_rmem_tensor((_MLA_OUTPUT_FRAGMENTS_PER_LANE,), Float32)
        for frag_idx in cutlass.range_constexpr(_MLA_OUTPUT_FRAGMENTS_PER_LANE):
            acc[frag_idx] = Float32(0.0)

        softmax_m = Float32(-Float32.inf)
        softmax_d = Float32(0.0)
        softmax_scale = Float32(sm_scale[Int32(0)])
        token_count = Int32(page_table_1.shape[1])
        num_kv = Int32(kv_nope_q.shape[0])

        token_pos = Int32(0)
        while token_pos < token_count:
            token_idx = page_table_1[q_idx, token_pos]
            token_pos += Int32(1)
            if token_idx >= Int32(0) and token_idx < num_kv:
                score = Float32(0.0)
                for group_idx in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
                    k0, k1, k2, k3 = _load_scaled_fp8x4(
                        kv_nope_q,
                        kv_scales,
                        token_idx,
                        group_idx,
                        lane,
                    )
                    q_base = Int32(group_idx * _MLA_GROUP_SIZE) + lane * Int32(4)
                    score = Float32(score + Float32(q_all[q_idx, head_idx, q_base + Int32(0)]) * k0)
                    score = Float32(score + Float32(q_all[q_idx, head_idx, q_base + Int32(1)]) * k1)
                    score = Float32(score + Float32(q_all[q_idx, head_idx, q_base + Int32(2)]) * k2)
                    score = Float32(score + Float32(q_all[q_idx, head_idx, q_base + Int32(3)]) * k3)

                rope_base = _MLA_NOPE_DIM + lane * Int32(2)
                score = Float32(
                    score
                    + Float32(q_all[q_idx, head_idx, rope_base + Int32(0)])
                    * Float32(kv_rope[token_idx, lane * Int32(2) + Int32(0)])
                )
                score = Float32(
                    score
                    + Float32(q_all[q_idx, head_idx, rope_base + Int32(1)])
                    * Float32(kv_rope[token_idx, lane * Int32(2) + Int32(1)])
                )
                score = Float32(_warp_allreduce_sum(score) * softmax_scale)

                new_m = attention_utils.fmax(softmax_m, score)
                alpha = (
                    Float32(0.0)
                    if softmax_d == Float32(0.0)
                    else Float32(cute.math.exp(softmax_m - new_m, fastmath=False))
                )
                prob = Float32(cute.math.exp(score - new_m, fastmath=False))
                softmax_d = Float32(softmax_d * alpha + prob)

                for group_idx in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
                    v0, v1, v2, v3 = _load_scaled_fp8x4(
                        kv_nope_q,
                        kv_scales,
                        token_idx,
                        group_idx,
                        lane,
                    )
                    frag_base = group_idx * 4
                    acc[frag_base + 0] = Float32(acc[frag_base + 0] * alpha + prob * v0)
                    acc[frag_base + 1] = Float32(acc[frag_base + 1] * alpha + prob * v1)
                    acc[frag_base + 2] = Float32(acc[frag_base + 2] * alpha + prob * v2)
                    acc[frag_base + 3] = Float32(acc[frag_base + 3] * alpha + prob * v3)
                softmax_m = Float32(new_m)

        inv_d = (
            Float32(0.0)
            if softmax_d == Float32(0.0)
            else Float32(1.0) / softmax_d
        )
        for group_idx in cutlass.range_constexpr(_MLA_SCALE_GROUPS):
            out_base = Int32(group_idx * _MLA_GROUP_SIZE) + lane * Int32(4)
            frag_base = group_idx * 4
            output[q_idx, head_idx, out_base + Int32(0)] = Float32(acc[frag_base + 0] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(1)] = Float32(acc[frag_base + 1] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(2)] = Float32(acc[frag_base + 2] * inv_d).to(
                output.element_type
            )
            output[q_idx, head_idx, out_base + Int32(3)] = Float32(acc[frag_base + 3] * inv_d).to(
                output.element_type
            )


@lru_cache(maxsize=8)
def _build_sparse_mla_kernel(traits: SparseMLATraits) -> SparseMLAKernel:
    del traits
    return SparseMLAKernel()


def clear_sparse_mla_kernel_cache() -> None:
    _build_sparse_mla_kernel.cache_clear()


def supports_sparse_mla_kernel(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    v_head_dim: int,
) -> bool:
    if os.environ.get("B12X_MLA_FORCE_REFERENCE", "0") == "1":
        return False
    return (
        select_sparse_mla_traits(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            output_dtype=q_all.dtype,
            v_head_dim=v_head_dim,
        )
        is not None
    )


def run_sparse_mla_kernel(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float | torch.Tensor,
    output: torch.Tensor,
) -> None:
    traits = select_sparse_mla_traits(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        output_dtype=output.dtype,
        v_head_dim=output.shape[-1],
    )
    if traits is None:
        raise ValueError("sparse MLA kernel only supports the exact CUDA GLM-5.1 contract")

    kv_rows = kv_cache[:, 0, :]
    kv_nope_q = kv_rows[:, :_MLA_NOPE_DIM]
    kv_scales = kv_rows[:, _MLA_NOPE_DIM : _MLA_NOPE_DIM + _MLA_SCALE_GROUPS * 4].view(torch.float32)
    kv_rope = kv_rows[:, _MLA_NOPE_DIM + _MLA_SCALE_GROUPS * 4 :].view(torch.bfloat16)
    if isinstance(sm_scale, torch.Tensor):
        sm_scale_tensor = sm_scale
    else:
        sm_scale_tensor = torch.tensor([sm_scale], dtype=torch.float32, device=q_all.device)
    if sm_scale_tensor.shape != (1,) or sm_scale_tensor.dtype != torch.float32:
        raise ValueError("sm_scale tensor must have shape (1,) and dtype float32")
    if sm_scale_tensor.device != q_all.device:
        raise ValueError("sm_scale tensor must be on the same device as q_all")

    kernel = _build_sparse_mla_kernel(traits)
    args = (
        _to_kernel_tensor(q_all, _torch_to_cutlass_dtype(q_all.dtype)),
        _to_kernel_tensor(kv_nope_q, cutlass.Uint8),
        _to_kernel_tensor(kv_scales, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(kv_rope, cutlass.BFloat16),
        _to_kernel_tensor(page_table_1, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(sm_scale_tensor, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(output, _torch_to_cutlass_dtype(output.dtype)),
        current_cuda_stream(),
    )
    cache_key = (
        _tensor_meta_key(q_all),
        _tensor_meta_key(kv_nope_q),
        _tensor_meta_key(kv_scales),
        _tensor_meta_key(kv_rope),
        _tensor_meta_key(page_table_1),
        _tensor_meta_key(output),
        traits,
        str(q_all.dtype),
        str(output.dtype),
    )
    _run_cached_host_launcher(kernel, cache_key, args)
