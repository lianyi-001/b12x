"""Benchmark the reusable BF16->FP4 TMA quantization kernel module."""

import pathlib
import statistics as _stats
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from b12x.quantization import allocate_bf16_to_fp4_tma_outputs, compile_bf16_to_fp4_tma

_M = int(sys.argv[sys.argv.index("--M") + 1]) if "--M" in sys.argv else 128
_K = int(sys.argv[sys.argv.index("--K") + 1]) if "--K" in sys.argv else 128
_dev = torch.device("cuda")
torch.manual_seed(42)
_bf16 = torch.randn(_M, _K, dtype=torch.bfloat16, device=_dev)
_gs = torch.tensor([1.0], dtype=torch.float32, device=_dev)
_rp = ((_M + 127) // 128) * 128
_csf = ((_K // 16 + 3) // 4) * 4
_in = _bf16 if _rp == _M and _bf16.is_contiguous() else torch.zeros((_rp, _K), dtype=torch.bfloat16, device=_dev)
if _rp != _M:
    _in[:_M].copy_(_bf16)
_out = allocate_bf16_to_fp4_tma_outputs(_M, _K, device=_dev)
compiled = compile_bf16_to_fp4_tma(_rp, _K)
print("Compiled OK")


def _launch():
    compiled(_in, _gs, _out.packed_a_flat, _out.scale_flat)


for _ in range(3): _launch()
torch.cuda.synchronize()
_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(_graph): _launch()
torch.cuda.synchronize()
_w, _n = 10, 50
for _ in range(_w): _graph.replay()
torch.cuda.synchronize()
_se=[torch.cuda.Event(enable_timing=True) for _ in range(_n)]
_ee=[torch.cuda.Event(enable_timing=True) for _ in range(_n)]
for _i in range(_n): _se[_i].record(); _graph.replay(); _ee[_i].record()
torch.cuda.synchronize()
_t=[_se[_i].elapsed_time(_ee[_i]) for _i in range(_n)]
_med=_stats.median(_t)*1000; _mn=min(_t)*1000
_rd=_rp*_K*2; _wr=_rp*_K//2+_rp*_csf; _bw=(_rd+_wr)/(_med*1e-6)/1e9
print(f"M={_M} K={_K}  graph replay median: {_med:.1f} us  (min {_mn:.1f})  BW: {_bw:.1f} GB/s")
