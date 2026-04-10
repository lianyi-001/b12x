#!/usr/bin/env python3
"""Benchmark graph-replayed sparse MLA through the public b12x API."""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from safetensors import safe_open

from b12x.attention.mla.reference import dense_mla_reference, pack_mla_kv_cache_reference
from b12x.integration.mla import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    MLAWorkspace,
    clear_mla_caches,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)

from benchmarks.common import require_sm120


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")
LAYER0_SHARD = MODEL_PATH / "model-00001-of-00084.safetensors"


@dataclass(frozen=True)
class GLMMLAConfig:
    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def sm_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


@dataclass(frozen=True)
class GLMMLAWeights:
    q_a_proj: torch.Tensor
    kv_a_proj_with_mqa: torch.Tensor
    q_b_proj: torch.Tensor
    q_a_layernorm: torch.Tensor
    kv_a_layernorm: torch.Tensor
    w_kc: torch.Tensor


def _require_glm_weights() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}")
    if not LAYER0_SHARD.exists():
        raise SystemExit(f"Layer-0 shard not found at {LAYER0_SHARD}")


@functools.lru_cache(maxsize=1)
def _load_glm_config() -> GLMMLAConfig:
    config = json.loads((MODEL_PATH / "config.json").read_text())
    return GLMMLAConfig(
        hidden_size=int(config["hidden_size"]),
        num_heads=int(config["num_attention_heads"]),
        q_lora_rank=int(config["q_lora_rank"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        v_head_dim=int(config["v_head_dim"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        rope_theta=float(config["rope_parameters"]["rope_theta"]),
    )


@functools.lru_cache(maxsize=1)
def _load_glm_layer0_cpu() -> GLMMLAWeights:
    cfg = _load_glm_config()
    keys = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
    ]
    with safe_open(str(LAYER0_SHARD), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key).contiguous() for key in keys}

    kv_b_proj = tensors["model.layers.0.self_attn.kv_b_proj.weight"]
    w_kc, _ = kv_b_proj.unflatten(
        0,
        (-1, cfg.qk_nope_head_dim + cfg.v_head_dim),
    ).split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=1)
    w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)

    return GLMMLAWeights(
        q_a_proj=tensors["model.layers.0.self_attn.q_a_proj.weight"],
        kv_a_proj_with_mqa=tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
        q_b_proj=tensors["model.layers.0.self_attn.q_b_proj.weight"],
        q_a_layernorm=tensors["model.layers.0.self_attn.q_a_layernorm.weight"],
        kv_a_layernorm=tensors["model.layers.0.self_attn.kv_a_layernorm.weight"],
        w_kc=w_kc,
    )


@functools.lru_cache(maxsize=4)
def _load_glm_layer0_cuda(device_index: int) -> tuple[GLMMLAConfig, GLMMLAWeights]:
    cfg = _load_glm_config()
    cpu = _load_glm_layer0_cpu()
    device = torch.device("cuda", device_index)
    return cfg, GLMMLAWeights(
        q_a_proj=cpu.q_a_proj.to(device=device),
        kv_a_proj_with_mqa=cpu.kv_a_proj_with_mqa.to(device=device),
        q_b_proj=cpu.q_b_proj.to(device=device),
        q_a_layernorm=cpu.q_a_layernorm.to(device=device),
        kv_a_layernorm=cpu.kv_a_layernorm.to(device=device),
        w_kc=cpu.w_kc.to(device=device),
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.to(torch.float32)
    inv_rms = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms).to(x.dtype) * weight


def _rope_interleaved(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(half, device=x.device, dtype=torch.float32)
            / half
        )
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = freqs.cos().view(x.shape[0], 1, half)
    sin = freqs.sin().view(x.shape[0], 1, half)

    x_pairs = x.to(torch.float32).reshape(x.shape[0], x.shape[1], half, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]
    rotated = torch.empty_like(x_pairs)
    rotated[..., 0] = even * cos - odd * sin
    rotated[..., 1] = even * sin + odd * cos
    return rotated.reshape_as(x).to(x.dtype)


def _make_glm_case(
    *,
    cache_len: int,
    q_len: int,
    seed: int,
    device: torch.device,
) -> tuple[GLMMLAConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg, weights = _load_glm_layer0_cuda(device.index or 0)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden_states = torch.randn(
        (cache_len, cfg.hidden_size),
        generator=gen,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    hidden_states /= 4
    positions = torch.arange(cache_len, device=device, dtype=torch.long)

    q_lora = F.linear(hidden_states, weights.q_a_proj)
    latent = F.linear(hidden_states, weights.kv_a_proj_with_mqa)
    q_norm = _rms_norm(q_lora, weights.q_a_layernorm, cfg.rms_norm_eps)
    k_nope = _rms_norm(
        latent[:, : cfg.kv_lora_rank],
        weights.kv_a_layernorm,
        cfg.rms_norm_eps,
    ).unsqueeze(1)
    k_rope = _rope_interleaved(
        latent[:, cfg.kv_lora_rank :].unsqueeze(1),
        positions,
        cfg.rope_theta,
    )

    q = F.linear(q_norm, weights.q_b_proj).view(
        cache_len,
        cfg.num_heads,
        cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
    )
    q_nope, q_rope = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), weights.w_kc).transpose(0, 1)
    q_rope = _rope_interleaved(q_rope, positions, cfg.rope_theta)
    q_all = torch.cat([q_nope_out[-q_len:], q_rope[-q_len:]], dim=-1).contiguous()
    return cfg, q_all, k_nope, k_rope


def _compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a - b).to(torch.float32)
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return diff.abs().max().item(), torch.sqrt(diff.square().mean()).item(), cos


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_page_table(
    *,
    cache_len: int,
    q_len: int,
    width: int,
    valid_per_row: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if valid_per_row > width:
        raise ValueError("valid_per_row cannot exceed width")
    if width <= 0 or valid_per_row <= 0:
        raise ValueError("width and valid_per_row must be positive")
    if width == cache_len and valid_per_row == cache_len:
        return torch.arange(cache_len, dtype=torch.int32, device=device).repeat(q_len, 1)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    rows = []
    for _ in range(q_len):
        valid = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[:valid_per_row]
        valid = valid.to(torch.int32)
        padded = torch.full((width,), -1, dtype=torch.int32)
        padded[:valid_per_row] = valid
        rows.append(padded)
    return torch.stack(rows, dim=0).to(device=device)


def _make_workspace(
    *,
    mode: str,
    device: torch.device,
    max_total_q: int,
    max_batch: int,
    topk: int,
    cfg: GLMMLAConfig,
) -> MLAWorkspace:
    return MLAWorkspace.for_fixed_capacity(
        mode=mode,
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=cfg.num_heads,
        head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        v_head_dim=cfg.kv_lora_rank,
        topk=topk,
        max_total_q=max_total_q,
        max_batch=max_batch,
    )


def _capture_graph(fn, *, warmup: int) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _bench_graph(graph: torch.cuda.CUDAGraph, *, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for idx in range(replays):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _run_case(
    *,
    mode: str,
    cache_len: int,
    q_len: int,
    width: int,
    valid_per_row: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
) -> None:
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=seed,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _make_page_table(
        cache_len=cache_len,
        q_len=q_len,
        width=width,
        valid_per_row=valid_per_row,
        seed=seed + 1,
        device=device,
    )

    if mode == "decode":
        cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
        metadata = MLASparseDecodeMetadata(
            page_table_1=page_table_1[:1],
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=cache_seqlens,
            max_seq_len_k=cache_len,
        )
        workspace = _make_workspace(
            mode="decode",
            device=device,
            max_total_q=1,
            max_batch=1,
            topk=width,
            cfg=cfg,
        )

        def run():
            return sparse_mla_decode_forward(
                q_all=q_all[:1],
                kv_cache=packed,
                metadata=metadata,
                workspace=workspace,
                sm_scale=cfg.sm_scale,
                v_head_dim=cfg.kv_lora_rank,
            )

        q_ref = q_all[:1]
        page_table_ref = page_table_1[:1]
    else:
        cache_seqlens = torch.full((q_len,), cache_len, dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(0, q_len + 1, dtype=torch.int32, device=device)
        metadata = MLASparseExtendMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=cache_seqlens,
            nsa_cu_seqlens_q=cu_seqlens,
            nsa_cu_seqlens_k=cu_seqlens,
            max_seq_len_q=1,
            max_seq_len_k=cache_len,
            mode="extend",
        )
        workspace = _make_workspace(
            mode="extend",
            device=device,
            max_total_q=q_len,
            max_batch=q_len,
            topk=width,
            cfg=cfg,
        )

        def run():
            return sparse_mla_extend_forward(
                q_all=q_all,
                kv_cache=packed,
                metadata=metadata,
                workspace=workspace,
                sm_scale=cfg.sm_scale,
                v_head_dim=cfg.kv_lora_rank,
            )

        q_ref = q_all
        page_table_ref = page_table_1

    clear_mla_caches()
    actual = run()
    expected = dense_mla_reference(
        q_all=q_ref,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_ref,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize()

    max_abs, rmse, cos = _compare(actual, expected)
    graph = _capture_graph(run, warmup=warmup)
    replay_us = _bench_graph(graph, replays=replays)

    print(
        json.dumps(
            {
                "mode": mode,
                "cache_len": cache_len,
                "q_len": q_len,
                "width": width,
                "valid_per_row": valid_per_row,
                "median_us": statistics.median(replay_us),
                "mean_us": statistics.fmean(replay_us),
                "min_us": min(replay_us),
                "max_us": max(replay_us),
                "replays": replays,
                "sanity": {
                    "max_abs": max_abs,
                    "rmse": rmse,
                    "cos": cos,
                },
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "extend", "both"), default="both")
    parser.add_argument("--decode-cache-lens", default="63,64,65,127,128,129")
    parser.add_argument("--extend-cache-lens", default="63,64,65,127,128,129")
    parser.add_argument("--extend-q-len", type=int, default=5)
    parser.add_argument("--width", type=int, default=0, help="0 means dense width=cache_len")
    parser.add_argument(
        "--valid-per-row",
        type=int,
        default=0,
        help="0 means valid_per_row=width, otherwise generate sparse padded rows",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--seed", type=int, default=70_000)
    args = parser.parse_args()

    device = require_sm120()
    _require_glm_weights()

    if args.mode in ("decode", "both"):
        for case_idx, cache_len in enumerate(_parse_csv_ints(args.decode_cache_lens)):
            width = cache_len if args.width == 0 else args.width
            valid_per_row = width if args.valid_per_row == 0 else args.valid_per_row
            _run_case(
                mode="decode",
                cache_len=cache_len,
                q_len=1,
                width=width,
                valid_per_row=valid_per_row,
                warmup=args.warmup,
                replays=args.replays,
                seed=args.seed + case_idx * 100,
                device=device,
            )

    if args.mode in ("extend", "both"):
        for case_idx, cache_len in enumerate(_parse_csv_ints(args.extend_cache_lens)):
            width = cache_len if args.width == 0 else args.width
            valid_per_row = width if args.valid_per_row == 0 else args.valid_per_row
            _run_case(
                mode="extend",
                cache_len=cache_len,
                q_len=args.extend_q_len,
                width=width,
                valid_per_row=valid_per_row,
                warmup=args.warmup,
                replays=args.replays,
                seed=args.seed + 10_000 + case_idx * 100,
                device=device,
            )


if __name__ == "__main__":
    main()
