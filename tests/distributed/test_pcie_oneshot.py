from __future__ import annotations

import pytest
import torch

from b12x.distributed.pcie_oneshot import (
    PCIeOneshotAllReduce,
    _compute_crossover_size,
    parse_pcie_oneshot_max_size,
)


class _FakeExt:
    def __init__(self):
        self.init_calls = []
        self.register_pcie_buffers_calls = []
        self.register_buffer_calls = []
        self.all_reduce_calls = []
        self.dispose_calls = []
        self.register_graph_buffers_calls = []
        self.handle_bytes = [1, 2, 3]
        self.offsets = [0, 64]

    def init_custom_ar(self, signal_ptrs, rank_data, rank):
        self.init_calls.append((tuple(signal_ptrs), rank_data.device.type, rank))
        return 12345

    def register_pcie_buffers(self, ptr, ptrs0, ptrs1):
        self.register_pcie_buffers_calls.append((ptr, tuple(ptrs0), tuple(ptrs1)))

    def register_buffer(self, ptr, peer_input_ptrs):
        self.register_buffer_calls.append((ptr, tuple(peer_input_ptrs)))

    def all_reduce(self, ptr, inp, out, reg_buffer, reg_buffer_bytes):
        self.all_reduce_calls.append((ptr, int(inp.data_ptr()), int(out.data_ptr()), reg_buffer, reg_buffer_bytes))
        out.copy_(inp)

    def dispose(self, ptr):
        self.dispose_calls.append(ptr)

    def meta_size(self):
        return 256

    def get_graph_buffer_ipc_meta(self, ptr):
        return list(self.handle_bytes), list(self.offsets)

    def register_graph_buffers(self, ptr, handles, offsets):
        self.register_graph_buffers_calls.append((ptr, handles, offsets))


def _make_runtime(*, rank=0, world_size=2, exchange_group=None, max_size=8 * 1024 * 1024):
    return PCIeOneshotAllReduce(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        exchange_group=exchange_group,
        max_size=max_size,
        ext_module=_FakeExt(),
    )


def test_parse_pcie_oneshot_max_size_accepts_auto_and_suffixes():
    assert parse_pcie_oneshot_max_size(None) is None
    assert parse_pcie_oneshot_max_size("auto") is None
    assert parse_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert parse_pcie_oneshot_max_size("2m") == 2 * 1024 * 1024
    assert parse_pcie_oneshot_max_size(4096) == 4096


def test_compute_crossover_size_runs_fine_sweep():
    seen_sizes = []

    def benchmark(size_bytes: int) -> tuple[float, float]:
        seen_sizes.append(size_bytes)
        if size_bytes <= 48 * 1024:
            return 1.0, 2.0
        return 3.0, 2.0

    crossover, results = _compute_crossover_size(
        benchmark,
        ceiling_bytes=64 * 1024,
        fine_step_bytes=8 * 1024,
    )

    assert crossover == 48 * 1024
    assert 40 * 1024 in seen_sizes
    assert 48 * 1024 in seen_sizes
    assert 56 * 1024 in seen_sizes
    assert results[-1].size_bytes == 64 * 1024


def test_register_buffer_is_idempotent_for_same_mapping():
    runtime = _make_runtime()
    ext = runtime._ext

    runtime.register_buffer((111, 222))
    runtime.register_buffer((111, 222))

    assert ext.register_buffer_calls == [(12345, (111, 222))]


def test_register_buffer_rejects_mismatched_mapping_for_same_local_ptr():
    runtime = _make_runtime()

    runtime.register_buffer((111, 222))

    with pytest.raises(ValueError, match="already registered"):
        runtime.register_buffer((111, 333))


def test_all_reduce_registers_explicit_peer_ptrs_once():
    runtime = _make_runtime()
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out0 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))
    out1 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))

    assert torch.equal(out0, inp)
    assert torch.equal(out1, inp)
    assert ext.register_buffer_calls == [(12345, (inp.data_ptr(), 222))]
    assert len(ext.all_reduce_calls) == 2


def test_all_reduce_requires_registration_without_eager_buffers():
    runtime = _make_runtime()
    inp = torch.arange(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="peer_input_ptrs are required"):
        runtime.all_reduce(inp)


def test_should_allreduce_checks_device_dtype_size_alignment_and_contiguity():
    runtime = _make_runtime(max_size=16)

    good = torch.arange(8, dtype=torch.bfloat16)
    assert runtime.should_allreduce(good) is True
    assert runtime.should_allreduce(torch.arange(4, dtype=torch.int32)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(7, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)[::2]) is False


def test_graph_buffer_api_exposes_explicit_registration_hooks():
    runtime = _make_runtime()
    ext = runtime._ext

    assert runtime.get_graph_buffer_ipc_meta() == ([1, 2, 3], [0, 64])

    runtime.register_graph_buffers_from_ranks(
        ([1, 2, 3], [4, 5, 6]),
        ([0, 64], [8, 72]),
    )

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [4, 5, 6]], [[0, 64], [8, 72]])
    ]


def test_register_graph_buffers_uses_exchange_group_broadcast(monkeypatch):
    remote_meta = {
        0: ([1, 2, 3], [0, 64]),
        1: ([9, 8, 7], [16, 80]),
    }

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda group=None: [0, 1])

    def fake_broadcast(object_list, src, group=None, device=None):
        object_list[0] = remote_meta[src]

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)

    runtime = _make_runtime(exchange_group=object())
    ext = runtime._ext
    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [9, 8, 7]], [[0, 64], [16, 80]])
    ]


def test_capture_registers_graph_buffers_after_context(monkeypatch):
    runtime = _make_runtime(exchange_group=object())
    calls = []

    monkeypatch.setattr(runtime, "register_graph_buffers", lambda: calls.append("registered"))

    with runtime.capture():
        pass

    assert calls == ["registered"]
