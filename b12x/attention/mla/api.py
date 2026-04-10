"""Sparse MLA API oriented around the NSA runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

import torch

from .kernel import (
    clear_sparse_mla_kernel_cache,
    run_sparse_mla_kernel,
    supports_sparse_mla_kernel,
)
from .reference import sparse_mla_reference
from .workspace import MLAWorkspace


@dataclass(frozen=True)
class MLASparseDecodeMetadata:
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    max_seq_len_k: int


@dataclass(frozen=True)
class MLASparseExtendMetadata:
    page_table_1: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    nsa_cache_seqlens_int32: torch.Tensor
    nsa_cu_seqlens_q: torch.Tensor
    nsa_cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int
    mode: Literal["extend", "verify", "target_verify", "draft_extend"] = "extend"


def clear_mla_caches() -> None:
    """Clear any cached MLA runtime state."""
    clear_sparse_mla_kernel_cache()


def sparse_mla_decode_forward(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: MLASparseDecodeMetadata,
    workspace: MLAWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    workspace.prepare_decode(
        metadata.page_table_1,
        metadata.cache_seqlens_int32,
        metadata.nsa_cache_seqlens_int32,
    )
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def sparse_mla_extend_forward(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: MLASparseExtendMetadata,
    workspace: MLAWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    workspace.prepare_extend(
        metadata.page_table_1,
        metadata.cache_seqlens_int32,
        metadata.nsa_cache_seqlens_int32,
    )
    return _run_sparse_mla(
        q_all=q_all,
        kv_cache=kv_cache,
        workspace=workspace,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
    )


def _run_sparse_mla(
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace: MLAWorkspace,
    sm_scale: float,
    v_head_dim: int,
) -> torch.Tensor:
    page_table_1 = workspace.page_table_1
    if page_table_1 is None:
        raise RuntimeError("workspace metadata is not prepared")
    if q_all.ndim != 3:
        raise ValueError(f"q_all must be rank-3, got {tuple(q_all.shape)}")
    if kv_cache.ndim != 3:
        raise ValueError(f"kv_cache must be rank-3, got {tuple(kv_cache.shape)}")
    if q_all.device != workspace.device:
        raise ValueError(
            f"q_all device {q_all.device} does not match workspace device {workspace.device}"
        )
    if kv_cache.device != workspace.device:
        raise ValueError(
            f"kv_cache device {kv_cache.device} does not match workspace device {workspace.device}"
        )
    if page_table_1.device != workspace.device:
        raise ValueError(
            f"page_table_1 device {page_table_1.device} does not match workspace device {workspace.device}"
        )
    if q_all.dtype != workspace.dtype:
        raise ValueError(f"q_all dtype {q_all.dtype} does not match workspace dtype {workspace.dtype}")
    if kv_cache.dtype != workspace.kv_dtype:
        raise ValueError(
            f"kv_cache dtype {kv_cache.dtype} does not match workspace kv_dtype {workspace.kv_dtype}"
        )
    if page_table_1.dtype != torch.int32:
        raise ValueError(f"page_table_1 must have dtype torch.int32, got {page_table_1.dtype}")
    if int(v_head_dim) != workspace.v_head_dim:
        raise ValueError(
            f"v_head_dim {v_head_dim} does not match workspace v_head_dim {workspace.v_head_dim}"
        )
    if q_all.shape[0] > workspace.max_total_q:
        raise ValueError(
            f"q_all rows {q_all.shape[0]} exceed workspace capacity {workspace.max_total_q}"
        )
    if q_all.shape[0] != page_table_1.shape[0]:
        raise ValueError(
            f"page_table_1 rows {page_table_1.shape[0]} do not match q_all rows {q_all.shape[0]}"
        )
    if page_table_1.shape[1] > workspace.topk:
        raise ValueError(
            f"page_table_1 width {page_table_1.shape[1]} exceeds workspace topk {workspace.topk}"
        )
    if q_all.shape[1] != workspace.num_q_heads:
        raise ValueError(
            f"q_all num_heads {q_all.shape[1]} does not match workspace num_q_heads {workspace.num_q_heads}"
        )
    if q_all.shape[-1] != workspace.head_dim:
        raise ValueError(
            f"q_all head_dim {q_all.shape[-1]} does not match workspace head_dim {workspace.head_dim}"
        )

    use_reference = os.environ.get("B12X_MLA_FORCE_REFERENCE", "0") == "1"
    if not use_reference and supports_sparse_mla_kernel(
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        v_head_dim=v_head_dim,
    ):
        sm_scale_tensor = workspace.sm_scale_tensor
        if (
            sm_scale_tensor is None
            or sm_scale_tensor.device != q_all.device
            or sm_scale_tensor.dtype != torch.float32
        ):
            sm_scale_tensor = torch.empty((1,), dtype=torch.float32, device=q_all.device)
            workspace.sm_scale_tensor = sm_scale_tensor
            workspace.sm_scale_value = None
        sm_scale_value = float(sm_scale)
        if workspace.sm_scale_value != sm_scale_value:
            sm_scale_tensor[0] = sm_scale_value
            workspace.sm_scale_value = sm_scale_value
        output = torch.empty(
            (q_all.shape[0], q_all.shape[1], v_head_dim),
            dtype=q_all.dtype,
            device=q_all.device,
        )
        run_sparse_mla_kernel(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            sm_scale=sm_scale_tensor,
            output=output,
        )
    else:
        output = sparse_mla_reference(
            q_all=q_all,
            kv_cache=kv_cache,
            page_table_1=page_table_1,
            sm_scale=sm_scale,
            v_head_dim=v_head_dim,
        )
    return output
