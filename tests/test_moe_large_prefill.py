"""Large-prefill MoE test matching the KLD regression scenario.

Exercises the dynamic kernel path (routed_rows > 640) with m=2048,
single-request prefill through a single MoE layer. Compares against
the NVFP4 reference oracle at tight tolerances.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.benchmark_moe import (
    MODEL_PATH,
    TP_RANK,
    TP_SIZE,
    ModelSpec,
    get_scale_contract_params,
    load_expert_weights,
    make_routed_inputs,
)
from b12x.integration.tp_moe import (
    allocate_tp_moe_workspace,
    b12x_moe_fp4,
    clear_tp_moe_caches,
)
from b12x.moe.fused.reference import compare_to_reference, moe_reference_nvfp4

from .helpers import require_sm120


def _require_model_weights() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


def _make_spec() -> ModelSpec:
    return ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )


@pytest.mark.parametrize("m", [128, 512, 2048])
def test_large_prefill_matches_oracle(m: int) -> None:
    """Single-request prefill at batch sizes that hit the dynamic kernel."""
    device = require_sm120()
    _require_model_weights()

    clear_tp_moe_caches()

    spec = _make_spec()
    weights = load_expert_weights(MODEL_PATH, spec, layer_idx=0)
    scale_params = get_scale_contract_params(weights, "shared")

    x, topk_ids, topk_weights = make_routed_inputs(spec, m, seed=m * 100, device=device)

    workspace = allocate_tp_moe_workspace(
        x,
        scale_params.a1_gscale,
        weights.w13_weight,
        scale_params.a2_gscale,
        weights.w2_weight,
        topk_ids,
        input_scales_static=True,
    )

    actual = b12x_moe_fp4(
        x,
        scale_params.a1_gscale,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        scale_params.g1_alphas,
        scale_params.a2_gscale,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        scale_params.g2_alphas,
        topk_weights,
        topk_ids,
        workspace=workspace,
        input_scales_static=True,
    )
    torch.cuda.synchronize(device)

    reference = moe_reference_nvfp4(
        x,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        scale_params.g1_alphas,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        scale_params.g2_alphas,
        scale_params.a1_gscale,
        scale_params.a2_gscale,
        topk_ids,
        topk_weights,
        spec.num_experts,
        spec.hidden_size,
        spec.I_tp,
    )

    metrics = compare_to_reference(actual, reference)
    print(f"\n  m={m}: routed_rows={m * spec.top_k} max_abs={metrics.max_abs:.6f} "
          f"rmse={metrics.rmse:.6f} cos={metrics.cos:.6f}")
    assert metrics.max_abs <= 1e-3, f"m={m}: max_abs={metrics.max_abs:.6f}"
    assert metrics.cos > 0.999, f"m={m}: cos={metrics.cos:.6f}"
