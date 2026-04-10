#!/usr/bin/env python3
"""Benchmark graph-replayed NSA top-k selection through the public b12x API."""

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

from b12x.attention.nsa_indexer.reference import sparse_nsa_index_reference
from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerExtendMetadata,
    clear_nsa_indexer_caches,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
)

from benchmarks.common import require_sm120


MODEL_PATH = pathlib.Path("/data/models/GLM-5.1-NVFP4")


@dataclass(frozen=True)
class GLMNSAConfig:
    num_heads: int
    head_dim: int = 128
    page_size: int = 64


@functools.lru_cache(maxsize=1)
def _load_glm_config() -> GLMNSAConfig:
    config_path = MODEL_PATH / "config.json"
    if not config_path.exists():
        raise SystemExit(f"GLM-5.1 config not found at {config_path}")
    config = json.loads(config_path.read_text())
    return GLMNSAConfig(num_heads=int(config["num_attention_heads"]))


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


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


def _make_q_and_weights(
    *,
    rows: int,
    cfg: GLMNSAConfig,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q_fp8 = (
        torch.randn((rows, cfg.num_heads, cfg.head_dim), generator=gen, dtype=torch.float32)
        .to(device=device)
        .div_(2.0)
    ).to(torch.float8_e4m3fn)
    weights = (
        torch.randn((rows, cfg.num_heads, 1), generator=gen, dtype=torch.float32)
        .to(device=device)
        .div_(cfg.num_heads**0.5)
    )
    return q_fp8, weights


def _make_index_k_cache(
    *,
    num_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    k = (
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32)
        .to(device=device)
        .div_(3.0)
    )
    return pack_nsa_index_k_cache_reference(k)


def _make_page_table(
    *,
    rows: int,
    width: int,
    valid_per_row: int,
    num_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if width <= 0:
        raise ValueError("width must be positive")
    if valid_per_row <= 0 or valid_per_row > width:
        raise ValueError("valid_per_row must be in [1, width]")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    out = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        ids = torch.randperm(num_tokens, generator=gen, dtype=torch.int64)[:valid_per_row].to(torch.int32)
        out[row, :valid_per_row] = ids
    return out.to(device=device)


def _assert_exact_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int((actual != expected).sum().item())
    raise AssertionError(
        f"NSA indexer correctness mismatch: {mismatch} differing entries, "
        f"actual[0]={actual[0].tolist()} expected[0]={expected[0].tolist()}"
    )


def _run_decode_case(
    *,
    cfg: GLMNSAConfig,
    q_rows: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
) -> None:
    q_fp8, weights = _make_q_and_weights(rows=q_rows, cfg=cfg, seed=seed, device=device)
    index_k_cache = _make_index_k_cache(num_tokens=cache_len, seed=seed + 1, device=device)
    valid_per_row = min(width, cache_len)
    page_table_1 = _make_page_table(
        rows=q_rows,
        width=width,
        valid_per_row=valid_per_row,
        num_tokens=cache_len,
        seed=seed + 2,
        device=device,
    )
    seqlens = torch.full((q_rows,), valid_per_row, dtype=torch.int32, device=device)
    metadata = NSAIndexerDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=seqlens,
    )

    def run():
        return sparse_nsa_index_decode_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual = run()
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.arange(q_rows, dtype=torch.int32, device=device),
        seqlens_per_query=seqlens,
        topk=topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    _assert_exact_match(actual, expected)

    graph = _capture_graph(run, warmup=warmup)
    replay_us = _bench_graph(graph, replays=replays)
    print(
        json.dumps(
            {
                "mode": "decode",
                "q_rows": q_rows,
                "cache_len": cache_len,
                "width": width,
                "topk": topk,
                "median_us": statistics.median(replay_us),
                "mean_us": statistics.fmean(replay_us),
                "min_us": min(replay_us),
                "max_us": max(replay_us),
                "replays": replays,
            }
        )
    )


def _run_extend_case(
    *,
    cfg: GLMNSAConfig,
    batch: int,
    q_len: int,
    cache_len: int,
    width: int,
    topk: int,
    warmup: int,
    replays: int,
    seed: int,
    device: torch.device,
) -> None:
    total_q = batch * q_len
    q_fp8, weights = _make_q_and_weights(rows=total_q, cfg=cfg, seed=seed, device=device)
    index_k_cache = _make_index_k_cache(num_tokens=cache_len, seed=seed + 1, device=device)
    valid_per_row = min(width, cache_len)
    page_table_1 = _make_page_table(
        rows=batch,
        width=width,
        valid_per_row=valid_per_row,
        num_tokens=cache_len,
        seed=seed + 2,
        device=device,
    )
    extend_lengths = [q_len] * batch
    seqlens_expanded = torch.full((total_q,), valid_per_row, dtype=torch.int32, device=device)
    metadata = NSAIndexerExtendMetadata(
        page_table_1=page_table_1,
        nsa_seqlens_expanded=seqlens_expanded,
        nsa_extend_seq_lens_list=extend_lengths,
    )

    def run():
        return sparse_nsa_index_extend_topk(
            q_fp8=q_fp8,
            weights=weights,
            index_k_cache=index_k_cache,
            metadata=metadata,
            topk=topk,
            page_size=cfg.page_size,
        )

    clear_nsa_indexer_caches()
    actual = run()
    expected = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.repeat_interleave(
            torch.arange(batch, dtype=torch.int32, device=device),
            torch.tensor(extend_lengths, dtype=torch.int32, device=device),
        ),
        seqlens_per_query=seqlens_expanded,
        topk=topk,
        page_size=cfg.page_size,
    )
    torch.cuda.synchronize()
    _assert_exact_match(actual[: expected.shape[0]], expected)

    graph = _capture_graph(run, warmup=warmup)
    replay_us = _bench_graph(graph, replays=replays)
    print(
        json.dumps(
            {
                "mode": "extend",
                "batch": batch,
                "q_len": q_len,
                "cache_len": cache_len,
                "width": width,
                "topk": topk,
                "median_us": statistics.median(replay_us),
                "mean_us": statistics.fmean(replay_us),
                "min_us": min(replay_us),
                "max_us": max(replay_us),
                "replays": replays,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("decode", "extend", "both"), default="both")
    parser.add_argument("--decode-rows", default="1,16")
    parser.add_argument("--extend-batches", default="8")
    parser.add_argument("--extend-q-lens", default="4")
    parser.add_argument("--cache-lens", default="2048,8192")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--seed", type=int, default=88_000)
    args = parser.parse_args()

    device = require_sm120()
    cfg = _load_glm_config()
    cache_lens = _parse_csv_ints(args.cache_lens)
    decode_rows = _parse_csv_ints(args.decode_rows)
    extend_batches = _parse_csv_ints(args.extend_batches)
    extend_q_lens = _parse_csv_ints(args.extend_q_lens)

    if args.topk > args.width:
        raise SystemExit("--topk must be <= --width")

    case_seed = args.seed
    if args.mode in ("decode", "both"):
        for cache_len in cache_lens:
            for q_rows in decode_rows:
                _run_decode_case(
                    cfg=cfg,
                    q_rows=q_rows,
                    cache_len=cache_len,
                    width=args.width,
                    topk=args.topk,
                    warmup=args.warmup,
                    replays=args.replays,
                    seed=case_seed,
                    device=device,
                )
                case_seed += 17
    if args.mode in ("extend", "both"):
        for cache_len in cache_lens:
            for batch in extend_batches:
                for q_len in extend_q_lens:
                    _run_extend_case(
                        cfg=cfg,
                        batch=batch,
                        q_len=q_len,
                        cache_len=cache_len,
                        width=args.width,
                        topk=args.topk,
                        warmup=args.warmup,
                        replays=args.replays,
                        seed=case_seed,
                        device=device,
                    )
                    case_seed += 17


if __name__ == "__main__":
    main()
