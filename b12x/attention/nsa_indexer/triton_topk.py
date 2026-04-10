from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_MAX_WIDTH_BLOCK = 2048
_MAX_TOPK_BLOCK = 2048


@triton.jit
def _float32_to_ordered_uint32(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits >> 31
    mask = tl.where(sign != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
    return bits ^ mask


@triton.jit
def _sparse_nsa_topk_ids_kernel(
    logits_ptr,
    page_table_1_ptr,
    query_row_to_batch_ptr,
    seqlens_per_query_ptr,
    output_ptr,
    logits_row_stride,
    page_table_row_stride,
    output_row_stride,
    width,
    gather_k,
    WIDTH_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, WIDTH_BLOCK)
    width_mask = offsets < width

    batch_row = tl.load(query_row_to_batch_ptr + pid).to(tl.int32)
    seq_len = tl.load(seqlens_per_query_ptr + pid).to(tl.int32)

    logits = tl.load(
        logits_ptr + pid * logits_row_stride + offsets,
        mask=width_mask,
        other=float("-inf"),
    ).to(tl.float32)
    token_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + offsets,
        mask=width_mask,
        other=-1,
    ).to(tl.int32)

    valid = width_mask & (offsets < seq_len) & (token_ids >= 0)
    ordered = _float32_to_ordered_uint32(logits)
    pos_rank = 0xFFFFFFFF - offsets.to(tl.uint32)
    keys = (ordered.to(tl.uint64) << 32) | pos_rank.to(tl.uint64)
    keys = tl.where(valid, keys, 0)

    top_keys = tl.topk(keys, k=TOPK_BLOCK)
    out_offsets = tl.arange(0, TOPK_BLOCK)
    store_mask = out_offsets < gather_k

    selected_pos = (0xFFFFFFFF - (top_keys & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    selected_valid = store_mask & (top_keys != 0) & (selected_pos < width)
    selected_ids = tl.load(
        page_table_1_ptr + batch_row * page_table_row_stride + selected_pos,
        mask=selected_valid,
        other=-1,
    ).to(tl.int32)
    tl.store(
        output_ptr + pid * output_row_stride + out_offsets,
        tl.where(selected_valid, selected_ids, -1),
        mask=store_mask,
    )


def supports_sparse_nsa_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    gather_k: int,
) -> bool:
    if os.environ.get("B12X_NSA_INDEXER_FORCE_TORCH_TOPK", "0") == "1":
        return False
    if logits.device.type != "cuda":
        return False
    if not (
        logits.device == page_table_1.device == query_row_to_batch.device == seqlens_per_query.device
    ):
        return False
    if logits.ndim != 2 or page_table_1.ndim != 2:
        return False
    if query_row_to_batch.ndim != 1 or seqlens_per_query.ndim != 1:
        return False
    if logits.shape[0] != query_row_to_batch.shape[0] or logits.shape[0] != seqlens_per_query.shape[0]:
        return False
    if logits.shape[1] != page_table_1.shape[1]:
        return False
    if logits.dtype != torch.float32:
        return False
    if page_table_1.dtype != torch.int32:
        return False
    if query_row_to_batch.dtype != torch.int32 or seqlens_per_query.dtype != torch.int32:
        return False
    if gather_k <= 0:
        return False
    width_block = triton.next_power_of_2(int(logits.shape[1]))
    topk_block = triton.next_power_of_2(int(gather_k))
    if width_block > _MAX_WIDTH_BLOCK or topk_block > _MAX_TOPK_BLOCK:
        return False
    if topk_block > width_block:
        return False
    return True


def run_sparse_nsa_topk_kernel(
    *,
    logits: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    output: torch.Tensor,
    gather_k: int,
) -> None:
    if not supports_sparse_nsa_topk_kernel(
        logits=logits,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        gather_k=gather_k,
    ):
        raise ValueError("sparse NSA Triton top-k kernel does not support this configuration")
    if output.ndim != 2 or output.shape[0] != logits.shape[0] or output.shape[1] < gather_k:
        raise ValueError(
            "output must have shape [rows, >= gather_k], got "
            f"{tuple(output.shape)} for rows={logits.shape[0]} gather_k={gather_k}"
        )
    if output.dtype != torch.int32 or output.device != logits.device:
        raise ValueError("output must be an int32 CUDA tensor on the same device as logits")
    if output.stride(-1) != 1:
        raise ValueError("output must be contiguous in the top-k dimension")

    rows, width = logits.shape
    if rows == 0 or width == 0 or gather_k == 0:
        return

    width_block = triton.next_power_of_2(int(width))
    topk_block = triton.next_power_of_2(int(gather_k))
    num_warps = 4 if width_block <= 512 else 8

    _sparse_nsa_topk_ids_kernel[(rows,)](
        logits,
        page_table_1,
        query_row_to_batch,
        seqlens_per_query,
        output,
        logits.stride(0),
        page_table_1.stride(0),
        output.stride(0),
        width,
        gather_k,
        WIDTH_BLOCK=width_block,
        TOPK_BLOCK=topk_block,
        num_warps=num_warps,
    )
