from __future__ import annotations

import torch

from b12x.attention.nsa_indexer.reference import (
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_reference,
    unpack_nsa_index_k_cache_reference,
)


def _manual_index_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    query_row_to_batch: torch.Tensor,
    seqlens_per_query: torch.Tensor,
    topk: int,
    page_size: int = 64,
) -> torch.Tensor:
    num_queries = q_fp8.shape[0]
    output = torch.full((num_queries, topk), -1, dtype=torch.int32, device=q_fp8.device)

    data_bytes = page_size * 128
    cache = index_k_cache.contiguous()
    for query_row in range(int(query_row_to_batch.numel())):
        batch_row = int(query_row_to_batch[query_row].item())
        seq_len = int(seqlens_per_query[query_row].item())
        scores: list[torch.Tensor] = []
        token_ids: list[int] = []
        for pos in range(min(seq_len, page_table_1.shape[1])):
            token_id = int(page_table_1[batch_row, pos].item())
            if token_id < 0:
                continue
            page_idx = token_id // page_size
            slot_idx = token_id % page_size
            quant = (
                cache[page_idx, slot_idx * 128 : (slot_idx + 1) * 128]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .to(torch.float32)
            )
            scale_offset = data_bytes + slot_idx * 4
            scale = cache[page_idx, scale_offset : scale_offset + 4].contiguous().view(torch.float32)[0]
            logits = torch.matmul(q_fp8[query_row].to(torch.float32), quant)
            score = (torch.relu(logits) * weights[query_row].to(torch.float32)).sum() * scale
            scores.append(score)
            token_ids.append(token_id)
        if not scores:
            continue
        score_tensor = torch.stack(scores)
        token_tensor = torch.tensor(token_ids, dtype=torch.int32, device=q_fp8.device)
        count = min(topk, score_tensor.numel())
        topk_pos = torch.argsort(score_tensor, descending=True, stable=True)[:count]
        output[query_row, :count] = token_tensor[topk_pos]
    return output


@torch.inference_mode()
def test_pack_nsa_index_k_cache_roundtrip_matches_input_for_odd_lengths() -> None:
    device = torch.device("cpu")
    for num_tokens in (63, 64, 65, 127, 128, 129):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(70_000 + num_tokens)
        k = torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 4
        packed = pack_nsa_index_k_cache_reference(k)
        unpacked = unpack_nsa_index_k_cache_reference(packed, num_tokens=num_tokens)
        max_abs = (unpacked - k).abs().max().item()
        rmse = torch.sqrt(((unpacked - k) ** 2).mean()).item()
        assert packed.shape == (((num_tokens + 63) // 64), 64 * (128 + 4))
        assert max_abs <= 0.08, f"num_tokens={num_tokens}: max_abs={max_abs:.6f}"
        assert rmse <= 0.008, f"num_tokens={num_tokens}: rmse={rmse:.6f}"


def test_sparse_nsa_index_reference_matches_manual_decode_topk() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_001)

    num_tokens = 96
    q_rows = 2
    num_heads = 4
    topk = 5

    k = torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    index_k_cache = pack_nsa_index_k_cache_reference(k)
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.full((2, 16), -1, dtype=torch.int32, device=device)
    page_table_1[0, :7] = torch.tensor([4, 9, 11, 18, 33, 40, 63], dtype=torch.int32)
    page_table_1[1, :9] = torch.tensor([2, 8, 15, 21, 45, 64, 65, 72, 80], dtype=torch.int32)
    query_row_to_batch = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqlens_per_query = torch.tensor([7, 9], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
    )
    expected = _manual_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
    )

    assert torch.equal(actual, expected)


def test_sparse_nsa_index_reference_handles_extend_expansion_and_padded_rows() -> None:
    device = torch.device("cpu")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(71_002)

    num_tokens = 130
    num_heads = 3
    topk = 4

    k = torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32, device=device) / 3
    index_k_cache = pack_nsa_index_k_cache_reference(k)
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32, device=device) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads), generator=gen, dtype=torch.float32, device=device)
    page_table_1 = torch.full((2, 12), -1, dtype=torch.int32, device=device)
    page_table_1[0, :8] = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.int32)
    page_table_1[1, :10] = torch.tensor([64, 66, 68, 70, 72, 74, 76, 78, 80, 82], dtype=torch.int32)
    query_row_to_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=device)
    seqlens_per_query = torch.tensor([7, 8, 8, 9, 10], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
    )

    expected = _manual_index_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=query_row_to_batch,
        seqlens_per_query=seqlens_per_query,
        topk=topk,
    )

    assert torch.equal(actual[:5], expected[:5])
    assert torch.equal(actual[5], torch.full((topk,), -1, dtype=torch.int32))


def test_sparse_nsa_index_reference_prefers_earlier_positions_on_ties() -> None:
    device = torch.device("cpu")
    num_tokens = 32
    num_heads = 3
    topk = 4

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), dtype=torch.float32, device=device) / 3
    )
    q_fp8 = torch.zeros((1, num_heads, 128), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((1, num_heads), dtype=torch.float32, device=device)
    page_table_1 = torch.full((1, 8), -1, dtype=torch.int32, device=device)
    page_table_1[0, :6] = torch.tensor([19, 7, 23, 5, 11, 13], dtype=torch.int32, device=device)

    actual = sparse_nsa_index_reference(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        page_table_1=page_table_1,
        query_row_to_batch=torch.tensor([0], dtype=torch.int32, device=device),
        seqlens_per_query=torch.tensor([6], dtype=torch.int32, device=device),
        topk=topk,
    )

    assert torch.equal(actual[0], torch.tensor([19, 7, 23, 5], dtype=torch.int32, device=device))
