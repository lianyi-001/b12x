from __future__ import annotations

import statistics

import torch

from b12x.cute.fp4 import quantize_grouped_nvfp4_torch


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def require_sm120() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to run b12x benchmarks")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise SystemExit(f"SM120 is required to run b12x benchmarks, got sm_{major}{minor}")
    return torch.device("cuda")


def bench_gpu_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms)


def compute_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    value = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax
    return torch.tensor([value], dtype=torch.float32, device=x.device)


def compute_per_group_global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=(1, 2)).to(torch.float32)
    numerator = torch.full_like(amax, FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
    return torch.where(amax > 0, numerator / amax, torch.ones_like(amax))


def make_quantized_operand(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    scale: float = 0.25,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    source = torch.randn(shape, device="cuda", dtype=dtype) * scale
    row_counts = torch.full((shape[0],), shape[1], dtype=torch.int32, device=source.device)
    tensor_amax = source.abs().max().to(torch.float32)
    global_scale = torch.tensor(
        [FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax],
        dtype=torch.float32,
        device=source.device,
    )
    packed, scales = quantize_grouped_nvfp4_torch(source, row_counts, global_scale)
    return (packed, scales), global_scale
