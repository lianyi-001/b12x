#!/usr/bin/env python3
"""Print basic stats for dumped b12x live-call inputs."""

from __future__ import annotations

import argparse
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=pathlib.Path)
    args = parser.parse_args()

    for path in sorted(args.dump_dir.glob("b12x_reference_dump_layer*.pt")):
        dump = torch.load(path, map_location="cpu")
        x = dump["x"].float()
        topk_weights = dump["topk_weights"].float()
        a1 = dump["a1_gscale"].float()
        a2 = dump["a2_gscale"].float()
        print(
            f"{path.name}: layer={dump.get('layer_id')} m={x.shape[0]} "
            f"x_norm={float(x.norm()):.6f} topk_norm={float(topk_weights.norm()):.6f} "
            f"topk_max={float(topk_weights.abs().max()):.6f} "
            f"a1_max={float(a1.abs().max()):.9f} a2_max={float(a2.abs().max()):.9f}"
        )


if __name__ == "__main__":
    main()
