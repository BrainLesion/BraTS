"""Unit tests for the GPU-free gate."""

from __future__ import annotations

import subprocess

import pytest

from scripts.verify.gpu import GpuError, ensure_free, parse_device_ids


def _completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _runner(gpu_rows: str, compute_rows: str):
    def run(args, timeout):
        if any(arg.startswith("--query-gpu") for arg in args):
            return _completed(stdout=gpu_rows)
        return _completed(stdout=compute_rows)

    return run


def test_parse_device_ids():
    assert parse_device_ids("1") == [1]
    assert parse_device_ids("0, 1 ,") == [0, 1]


def test_parse_device_ids_invalid():
    with pytest.raises(GpuError):
        parse_device_ids("gpu0")


def test_ensure_free_when_idle():
    runner = _runner("0, GPU-aaa, 0, 10", "")
    assert ensure_free([0], runner=runner, wait_s=0) is True


def test_ensure_free_when_compute_process_running():
    runner = _runner("0, GPU-aaa, 0, 10", "GPU-aaa, 4242, 8000")
    assert ensure_free([0], runner=runner, wait_s=0) is False


def test_ensure_free_when_memory_used():
    runner = _runner("0, GPU-aaa, 0, 9000", "")
    assert ensure_free([0], runner=runner, wait_s=0) is False


def test_ensure_free_waits_then_succeeds():
    calls = {"n": 0}

    def runner(args, timeout):
        if any(arg.startswith("--query-gpu") for arg in args):
            return _completed(stdout="0, GPU-aaa, 0, 10")
        calls["n"] += 1
        busy = "GPU-aaa, 1, 8000" if calls["n"] == 1 else ""
        return _completed(stdout=busy)

    sleeps: list[float] = []
    assert (
        ensure_free([0], runner=runner, wait_s=100, poll_s=5, sleep=sleeps.append)
        is True
    )
    assert sleeps == [5]


def test_ensure_free_missing_nvidia_smi():
    def runner(args, timeout):
        raise FileNotFoundError

    with pytest.raises(GpuError):
        ensure_free([0], runner=runner)


def test_ensure_free_invalid_index():
    runner = _runner("0, GPU-aaa, 0, 10", "")
    with pytest.raises(GpuError):
        ensure_free([5], runner=runner, wait_s=0)
