#!/usr/bin/env python3
"""Print checkpoint-derived weight/blockscale summaries for one expert."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import benchmarks.benchmark_moe as bm
from benchmarks.benchmark_moe import MODEL_PATH, ModelSpec, load_expert_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--expert-id", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, default=0)
    args = parser.parse_args()

    cfg = json.loads((MODEL_PATH / "config.json").read_text())["text_config"]
    spec = ModelSpec(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["moe_intermediate_size"],
        num_experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
        tp_size=4,
        tp_rank=args.tp_rank,
    )
    bm.TP_RANK = args.tp_rank
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=args.layer_idx)

    eid = args.expert_id
    w1_u8_sum = int(weights.w13_weight[eid, :4, :16].to(torch.int32).sum().item())
    w2_u8_sum = int(weights.w2_weight[eid, :16, :4].to(torch.int32).sum().item())
    w1_bs_u8_sum = float(
        weights.w13_blockscale_swizzled[eid].view(torch.uint8)[:64].to(torch.float32).sum().item()
    )
    w2_bs_u8_sum = float(
        weights.w2_blockscale_swizzled[eid].view(torch.uint8)[:64].to(torch.float32).sum().item()
    )
    g1 = float(weights.g1_alphas[eid].item())
    g2 = float(weights.g2_alphas[eid].item())

    print(f"layer={args.layer_idx} expert={eid} tp_rank={args.tp_rank}")
    print(f"w1_u8_sum={w1_u8_sum}")
    print(f"w2_u8_sum={w2_u8_sum}")
    print(f"w1bs_u8_sum={w1_bs_u8_sum:.1f}")
    print(f"w2bs_u8_sum={w2_bs_u8_sum:.1f}")
    print(f"g1_alpha={g1:.9e}")
    print(f"g2_alpha={g2:.9e}")


if __name__ == "__main__":
    main()
