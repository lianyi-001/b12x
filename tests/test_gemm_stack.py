from __future__ import annotations

import pathlib
import sys

import pytest
import torch

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch
from b12x.cute.utils import convert_sf_from_mma_layout, get_num_sm
from b12x.gemm.dense import dense_gemm

_FLASHINFER_ROOT = pathlib.Path(__file__).resolve().parents[2] / "flashinfer"
if _FLASHINFER_ROOT.exists():
    sys.path.insert(0, str(_FLASHINFER_ROOT))

from flashinfer.gemm.gemm_base import CUDNN_AVAILABLE
from flashinfer.gemm import mm_fp4

from .helpers import require_sm120


def _require_cudnn_fp4() -> None:
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN Python bindings not installed")
    try:
        from flashinfer.gemm.gemm_base import _check_cudnn_fp4_availability
        _check_cudnn_fp4_availability()
    except RuntimeError as e:
        pytest.skip(f"cuDNN FP4 not available: {e}")


def _make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) / 4
    row_counts = torch.full((shape[0],), shape[1], dtype=torch.int32, device=source.device)
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale


@pytest.mark.parametrize("M,N,K", [
    (128, 128, 128),
    (256, 128, 128),
    (128, 256, 128),
    (128, 128, 256),
    (256, 256, 256),
    (256, 512, 128),
    (128, 256, 512),
    (512, 256, 256),
    (256, 256, 512),
])
@pytest.mark.parametrize("c_dtype_str", ["bfloat16", "float16"])
def test_dense_gemm_matches_flashinfer_cudnn(
    M: int, N: int, K: int, c_dtype_str: str,
) -> None:
    require_sm120()
    _require_cudnn_fp4()
    torch.manual_seed(42)

    lhs, lhs_scale = _make_quantized_operand((1, M, K), dtype=torch.bfloat16)
    rhs, rhs_scale = _make_quantized_operand((1, N, K), dtype=torch.bfloat16)
    alpha = (1.0 / (lhs_scale[0] * rhs_scale[0])).view(1)
    c_dtype = torch.bfloat16 if c_dtype_str == "bfloat16" else torch.float16

    dense_out = dense_gemm(
        lhs,
        rhs,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=c_dtype_str,
        sf_vec_size=16,
    )

    packed_a, sfa = lhs
    packed_b, sfb = rhs

    a_fp4 = packed_a[:, :, 0].contiguous()
    b_fp4 = packed_b[:, :, 0].contiguous()

    sfa_2d = convert_sf_from_mma_layout(sfa, m=M, k=K, num_groups=1)
    sfb_2d = convert_sf_from_mma_layout(sfb, m=N, k=K, num_groups=1)

    cudnn_out = mm_fp4(
        a_fp4,
        b_fp4.T,
        sfa_2d,
        sfb_2d.T,
        alpha,
        c_dtype,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cudnn",
        use_nvfp4=True,
    )

    torch.testing.assert_close(dense_out[:, :, 0], cudnn_out, rtol=0, atol=0)
