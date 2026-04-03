"""Shared b12x workspace helpers for the serving runtime."""

from __future__ import annotations

from typing import Any

import torch

_MOE_WORKSPACE_POOLS: dict[int, Any] = {}


def get_b12x_moe_workspace_pool(device: torch.device | str):
    """Return the process-local b12x TP MoE workspace pool for *device*.

    b12x's pool is already stream-aware internally, so one caller-owned pool
    per process/device can be reused across layers and CUDA graphs.
    """

    from b12x.integration.tp_moe import allocate_tp_moe_workspace_pool

    device = torch.device(device)
    if device.type == "cuda":
        device_idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        device_idx = -1
    pool = _MOE_WORKSPACE_POOLS.get(device_idx)
    if pool is None:
        pool = allocate_tp_moe_workspace_pool()
        _MOE_WORKSPACE_POOLS[device_idx] = pool
    return pool
