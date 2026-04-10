from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch

from b12x.integration.nsa_indexer import (
    NSAIndexerDecodeMetadata,
    NSAIndexerExtendMetadata,
    pack_nsa_index_k_cache_reference,
    sparse_nsa_index_decode_topk,
    sparse_nsa_index_extend_topk,
)


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_nsa_indexer():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa.nsa_indexer")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA indexer: {exc}")
    return module


class _FakePool:
    page_size = 64

    def __init__(self, index_k_cache: torch.Tensor):
        self._index_k_cache = index_k_cache

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._index_k_cache


class _FakeDecodeMode:
    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        del include_v2
        return False


class _FakeExtendMode:
    def is_decode_or_idle(self) -> bool:
        return False

    def is_extend_without_speculative(self) -> bool:
        return True


class _FakeAttnBackend:
    nsa_decode_impl = "b12x_mla"
    nsa_prefill_impl = "b12x_mla"


class _FakePagedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        seqlens_int32: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        self._page_table_1 = page_table_1
        self._seqlens_int32 = seqlens_int32
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_seqlens_int32(self) -> torch.Tensor:
        return self._seqlens_int32

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens


class _FakeRaggedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        self._page_table_1 = page_table_1
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens


def _make_fake_indexer(module, *, topk: int):
    class _FakeIndexer:
        index_topk = topk
        _use_b12x_mla_indexer = staticmethod(module.Indexer._use_b12x_mla_indexer)
        _get_b12x_paged_topk = module.Indexer._get_b12x_paged_topk
        _get_b12x_ragged_topk = module.Indexer._get_b12x_ragged_topk

    return _FakeIndexer()


def test_sglang_b12x_nsa_indexer_paged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_nsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_100)

    num_tokens = 96
    num_heads = 3
    topk = 4
    q_rows = 3

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1 = torch.full((q_rows, 8), -1, dtype=torch.int32)
    page_table_1[0, :5] = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
    page_table_1[1, :6] = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.int32)
    page_table_1[2, :7] = torch.tensor([20, 21, 22, 23, 24, 25, 26], dtype=torch.int32)
    seqlens = torch.tensor([5, 6, 7], dtype=torch.int32)

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1],
    )

    actual = module.Indexer._get_topk_paged(
        fake_indexer,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected = sparse_nsa_index_decode_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerDecodeMetadata(
            page_table_1=page_table_1,
            cache_seqlens_int32=seqlens,
        ),
        topk=topk,
    )

    assert torch.equal(actual, expected)


def test_sglang_b12x_nsa_indexer_ragged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_nsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_101)

    num_tokens = 128
    num_heads = 2
    topk = 5

    index_k_cache = pack_nsa_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1 = torch.full((2, 12), -1, dtype=torch.int32)
    page_table_1[0, :7] = torch.tensor([1, 3, 5, 7, 9, 11, 13], dtype=torch.int32)
    page_table_1[1, :9] = torch.tensor([64, 66, 68, 70, 72, 74, 76, 78, 80], dtype=torch.int32)
    seqlens_expanded = torch.tensor([6, 7, 7, 8, 9], dtype=torch.int32)
    extend_lens = [2, 3]

    fake_indexer = _make_fake_indexer(module, topk=topk)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeExtendMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(),
        },
    )()
    metadata = _FakeRaggedMetadata(
        page_table_1=page_table_1,
        seqlens_expanded=seqlens_expanded,
        extend_lens=extend_lens,
    )

    actual = module.Indexer._get_topk_ragged(
        fake_indexer,
        False,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected = sparse_nsa_index_extend_topk(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=NSAIndexerExtendMetadata(
            page_table_1=page_table_1,
            nsa_seqlens_expanded=seqlens_expanded,
            nsa_extend_seq_lens_list=extend_lens,
        ),
        topk=topk,
    )

    assert torch.equal(actual, expected)
