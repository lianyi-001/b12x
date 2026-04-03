"""Initialization-path tests for ServingEngine runtime policy wiring."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from serve.engine.serving import ServingEngine


class _FakeTokenizer:
    chat_template = None
    eos_token_id = None
    unk_token_id = -1

    def convert_tokens_to_ids(self, _name):
        return -1


class _FakePagePool:
    def __init__(self, *, num_pages, num_layers, kv_heads, head_dim, kv_dtype, device):
        del num_layers, kv_heads, head_dim, kv_dtype, device
        self.num_pages = num_pages
        self.page_size = 64
        self.k_cache = [torch.zeros(1)]
        self.v_cache = [torch.zeros(1)]

    @staticmethod
    def estimate_num_pages(*_args, **_kwargs):
        return 32


def test_serving_engine_passes_runtime_limits_to_model_runner(monkeypatch):
    cfg = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        vocab_size=16,
        layer_types=None,
    )
    fake_model = SimpleNamespace(config=cfg)
    runner_kwargs = {}

    class _FakeRunner:
        def __init__(self, model, kv_mgr, **kwargs):
            del model, kv_mgr
            runner_kwargs.update(kwargs)

        def warmup(self, *args, **kwargs):
            return None

        def capture_decode_graphs(self, *args, **kwargs):
            return None

        def compile_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr("serve.engine.serving.load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr("serve.engine.serving.AutoTokenizer.from_pretrained", lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr("serve.engine.serving._estimate_loaded_model_bytes", lambda model: 0)
    monkeypatch.setattr("serve.engine.serving.PagePool", _FakePagePool)
    monkeypatch.setattr("serve.engine.serving.PrefixCheckpointCache", lambda pool, state_arena=None: SimpleNamespace(pool=pool))
    monkeypatch.setattr("serve.engine.serving.start_startup_session", lambda: None)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (8 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr("serve.engine.runner.ModelRunner", _FakeRunner)

    engine = ServingEngine(
        "/tmp/fake-model",
        device="cpu",
        graph_batch_sizes=[],
        prefill_chunk_size=8192,
        max_running=256,
        max_prefill_tokens=16384,
    )

    assert runner_kwargs["max_batch_size"] == 256
    assert runner_kwargs["max_total_tokens"] == 16384
    assert engine.runtime_policy()["max_running"] == 256
