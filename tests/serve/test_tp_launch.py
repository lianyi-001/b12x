"""Tests for TP launch-side health monitoring helpers."""

from __future__ import annotations

from serve.tp.launch import TPFollowerMonitor


class _FakeProcess:
    def __init__(self, *, pid: int, alive: bool, exitcode: int | None):
        self.pid = pid
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


def test_tp_health_marks_clean_early_exit_as_fatal():
    monitor = TPFollowerMonitor(
        world_size=2,
        gpu_ids=[0, 1],
        followers=[_FakeProcess(pid=1234, alive=False, exitcode=0)],
    )

    health = monitor.health()
    assert health["fatal"] is True
    assert "exited unexpectedly" in health["summary"]
