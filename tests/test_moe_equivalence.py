"""Smoke test that the static TP MoE path returns a non-zero tensor."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_moe import MODEL_PATH, TP_RANK, TP_SIZE, ModelSpec, load_expert_weights


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"Requires SM120, got sm_{major}{minor}")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_moe_nonzero(m):
    """Validate `b12x_moe_fp4` produces non-zero output with real weights."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import _STATIC_KERNEL_CACHE, _STATE_CACHE, _WEIGHT_CACHE, b12x_moe_fp4

    _STATE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()

    device = torch.device("cuda")
    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )
    weights = load_expert_weights(MODEL_PATH, spec)

    torch.manual_seed(99)
    x = torch.randn(m, spec.hidden_size, dtype=torch.bfloat16, device=device)
    routing = torch.randn(m, spec.num_experts, device=device, dtype=torch.float32)
    topk_logits, topk_ids = torch.topk(routing, 10, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)

    out = b12x_moe_fp4(
        x,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights, topk_ids,
        implementation="static",
    )
    torch.cuda.synchronize()

    out_norm = out.float().norm().item()
    print(f"\nm={m}: out_norm={out_norm:.4f}, shape={out.shape}")
    assert out_norm > 0.01, f"m={m}: output is near-zero (norm={out_norm})"
    assert out.shape == (m, spec.hidden_size)


def _make_routed_inputs(spec: ModelSpec, m: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    x = torch.randn(m, spec.hidden_size, dtype=torch.bfloat16, device=device)
    routing = torch.randn(m, spec.num_experts, device=device, dtype=torch.float32)
    topk_logits, topk_ids = torch.topk(routing, 10, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1)
    return x, topk_ids, topk_weights


@pytest.mark.parametrize("m", [1, 2])
def test_moe_cuda_graph_replay_tracks_routing_updates(m):
    """Validate graph replay stays correct when routing contents change."""
    _skip_if_unavailable()

    from b12x.integration.tp_moe import _STATIC_KERNEL_CACHE, _STATE_CACHE, _WEIGHT_CACHE, b12x_moe_fp4

    _STATE_CACHE.clear()
    _WEIGHT_CACHE.clear()
    _STATIC_KERNEL_CACHE.clear()

    device = torch.device("cuda")
    spec = ModelSpec(
        hidden_size=4096,
        intermediate_size=1024,
        num_experts=512,
        top_k=10,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )
    weights = load_expert_weights(MODEL_PATH, spec)

    x0, topk_ids0, topk_weights0 = _make_routed_inputs(spec, m, seed=123, device=device)
    x_buf = x0.clone()
    topk_ids_buf = topk_ids0.clone()
    topk_weights_buf = topk_weights0.clone()
    graph_output = torch.empty_like(x_buf)

    # Compile once before capture; the replay check below is about routing safety.
    b12x_moe_fp4(
        x_buf,
        weights.w13_input_scale_per_expert,
        weights.w13_weight,
        weights.w13_blockscale_swizzled,
        weights.g1_alphas_per_expert,
        weights.w2_input_scale_per_expert,
        weights.w2_weight,
        weights.w2_blockscale_swizzled,
        weights.g2_alphas_per_expert,
        topk_weights_buf,
        topk_ids_buf,
        implementation="static",
        output=graph_output,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b12x_moe_fp4(
            x_buf,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights_buf,
            topk_ids_buf,
            implementation="static",
            output=graph_output,
        )

    for seed in (123, 456):
        x, topk_ids, topk_weights = _make_routed_inputs(spec, m, seed=seed, device=device)
        x_buf.copy_(x)
        topk_ids_buf.copy_(topk_ids)
        topk_weights_buf.copy_(topk_weights)

        graph.replay()
        torch.cuda.synchronize()
        replay_out = graph_output.clone()

        eager_out = b12x_moe_fp4(
            x,
            weights.w13_input_scale_per_expert,
            weights.w13_weight,
            weights.w13_blockscale_swizzled,
            weights.g1_alphas_per_expert,
            weights.w2_input_scale_per_expert,
            weights.w2_weight,
            weights.w2_blockscale_swizzled,
            weights.g2_alphas_per_expert,
            topk_weights,
            topk_ids,
            implementation="static",
        )
        torch.cuda.synchronize()

        diff = (replay_out.float() - eager_out.float()).abs()
        max_abs = diff.max().item()
        cos = F.cosine_similarity(
            replay_out.float().reshape(m, -1),
            eager_out.float().reshape(m, -1),
            dim=1,
        ).mean().item()

        assert max_abs < 5e-4, f"m={m} seed={seed}: max_abs={max_abs:.6f}"
        assert cos > 0.9999, f"m={m} seed={seed}: cos={cos:.6f}"
