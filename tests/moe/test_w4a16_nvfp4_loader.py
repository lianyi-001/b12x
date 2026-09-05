"""Bit-exact register packing for native NVFP4 weight loads."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import pytest
import torch

from b12x.moe._shared.kernels.w4a16.kernel import (
    MoEMicroKernelW4A16SmallMDirect,
    _pack_modelopt_words,
)


class _PackWords:
    @cute.jit
    def __call__(self, words: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(words, result).launch(grid=(words.shape[0] // 128, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, words: cute.Tensor, result: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        row = block * 128 + tid
        for byte_index in cutlass.range_constexpr(4):
            result[row, byte_index] = _pack_modelopt_words(
                words[row, 0], words[row, 1], words[row, 2], words[row, 3],
                cutlass.Uint32(byte_index),
            )


class _DecodeScaleBytes:
    def __init__(self):
        self.consumer = MoEMicroKernelW4A16SmallMDirect(
            activation="silu", fast_math=True, share_input_across_experts=False,
            share_expert_scales=True, single_token=False,
        )

    @cute.jit
    def __call__(self, codes: cute.Tensor, result: cute.Tensor, stream: cuda.CUstream):
        self.kernel(codes, result).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream,
        )

    @cute.kernel
    def kernel(self, codes: cute.Tensor, result: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        if tid < 127:
            result[tid] = self.consumer._scale_byte_to_f32(cutlass.Uint32(codes[tid]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modelopt_decode_preserves_all_finite_nonnegative_e4m3_scales() -> None:
    codes = torch.arange(127, device="cuda", dtype=torch.uint8)
    expected = codes.view(torch.float8_e4m3fn).float()
    result = torch.empty_like(expected)
    args = (
        from_dlpack(codes), from_dlpack(result),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    launch = cute.compile(_DecodeScaleBytes(), *args)
    launch(*args)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_modelopt_register_pack_preserves_every_fp4_code() -> None:
    generator = torch.Generator().manual_seed(13)
    random_words = torch.randint(0, 2**32, (768, 4), generator=generator, dtype=torch.int64)
    repeated_bytes = (torch.arange(256, dtype=torch.int64) * 0x01010101)[:, None].expand(256, 4)
    words_cpu = torch.cat((repeated_bytes, random_words))
    expected = torch.zeros((1024, 4), dtype=torch.int64)
    for byte_index in range(4):
        for source_index in range(4):
            byte = (words_cpu[:, source_index] >> (8 * byte_index)) & 255
            expected[:, byte_index] |= (byte & 15) << (4 * source_index)
            expected[:, byte_index] |= (byte >> 4) << (16 + 4 * source_index)

    words = words_cpu.to(device="cuda", dtype=torch.uint32)
    result = torch.empty((1024, 4), device="cuda", dtype=torch.uint32)
    args = (from_dlpack(words), from_dlpack(result), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    launch = cute.compile(_PackWords(), *args)
    launch(*args)
    torch.testing.assert_close(result.cpu().to(torch.int64), expected, rtol=0, atol=0)
