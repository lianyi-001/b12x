#!/usr/bin/env python3
"""Compare baseline vs dynamic_amax with adversarial activations."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from benchmarks.benchmark_moe import ModelSpec, load_expert_weights, make_routed_inputs
from b12x.moe.fused.reference import compare_to_reference, moe_reference_fp32_pure
from b12x.integration.tp_moe import b12x_moe_fp4, allocate_tp_moe_workspace, clear_tp_moe_caches

MODEL_PATH = pathlib.Path(os.environ.get("B12X_MODEL_PATH", "/data/models/Qwen3.5-397B-A17B-NVFP4"))
device = torch.device("cuda")

spec = ModelSpec(hidden_size=4096, intermediate_size=1024, num_experts=512, top_k=10, tp_size=4, tp_rank=0)
weights = load_expert_weights(MODEL_PATH, spec)
K = spec.hidden_size
I_tp = spec.intermediate_size // spec.tp_size

for spike_factor in [1.0, 10.0, 50.0, 200.0]:
    m = 4
    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=42, device=device)

    if spike_factor > 1.0:
        # Spike one K-tile per row — creates cross-tile dynamic range.
        spike_tile = 7
        x[:, spike_tile * 128 : (spike_tile + 1) * 128] *= spike_factor

    # Compute per-expert global scale from actual activations (not checkpoint).
    # This simulates a naive calibration: single amax over the whole expert.
    expert_amax = torch.zeros(spec.num_experts, dtype=torch.float32, device=device)
    for t in range(m):
        for k in range(topk_ids.shape[1]):
            eid = topk_ids[t, k].item()
            expert_amax[eid] = max(expert_amax[eid].item(), x[t].float().abs().max().item())
    # global_scale = amax / (6 * 448).
    computed_gs = (expert_amax / (6.0 * 448.0)).clamp(min=1e-12)
    # alpha = global_scale * weight_scale. We need to reconstruct weight_scale
    # from the checkpoint: alpha_checkpoint = checkpoint_gs * weight_scale.
    checkpoint_gs = weights.w13_input_scale_per_expert
    weight_scale = weights.g1_alphas_per_expert / checkpoint_gs
    computed_alpha = computed_gs * weight_scale

    oracle = moe_reference_fp32_pure(
        x, weights.w13_weight, weights.w13_blockscale_swizzled, weights.g1_alphas_per_expert,
        weights.w2_weight, weights.w2_blockscale_swizzled, weights.g2_alphas_per_expert,
        weights.w13_input_scale_per_expert, weights.w2_input_scale_per_expert,
        topk_ids, topk_weights, spec.num_experts, K, I_tp,
    ).to(torch.bfloat16)

    for label, amax_val in [("baseline", "0"), ("dynamic_amax", "1")]:
        os.environ["B12X_ENABLE_DYNAMIC_AMAX"] = amax_val
        clear_tp_moe_caches()
        ws = allocate_tp_moe_workspace(x, computed_gs, weights.w13_weight,
            weights.w2_input_scale_per_expert, weights.w2_weight, topk_ids,
            input_scales_static=False)
        out = b12x_moe_fp4(
            x, computed_gs,
            weights.w13_weight, weights.w13_blockscale_swizzled,
            computed_alpha,
            weights.w2_input_scale_per_expert,
            weights.w2_weight, weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights, topk_ids,
            workspace=ws,
        )
        torch.cuda.synchronize()
        met = compare_to_reference(out.float(), oracle.float())
        print(f"spike={spike_factor:5.1f}x {label:>13s}: max_abs={met.max_abs:.6f} rmse={met.rmse:.6f} cos={met.cos:.6f}")
    print()
