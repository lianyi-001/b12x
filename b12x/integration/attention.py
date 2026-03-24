"""Public paged-attention integration surface for the primary backend."""

from __future__ import annotations

from b12x.attention.paged.api import clear_paged_caches
from b12x.attention.paged.planner import infer_paged_mode as infer_paged_attention_mode
from b12x.attention.paged.workspace import PagedAttentionWorkspace


def clear_attention_caches() -> None:
    clear_paged_caches()


__all__ = [
    "PagedAttentionWorkspace",
    "clear_attention_caches",
    "infer_paged_attention_mode",
]
