"""Unit tests for registry preflight parsing."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.verify.registry import manifest_inspect


def _runner(returncode=0, stdout="", stderr=""):
    def run(args):
        return subprocess.CompletedProcess(
            args=args, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return run


def test_manifest_inspect_pullable_with_size():
    payload = json.dumps({"layers": [{"size": 1_000_000_000}, {"size": 500_000_000}]})
    result = manifest_inspect("brainles/example:latest", runner=_runner(stdout=payload))
    assert result.state == "pullable"
    assert result.pullable
    assert result.size_gb is not None
    assert result.size_gb == pytest.approx(1.5)


def test_manifest_inspect_not_found():
    result = manifest_inspect(
        "brainles/missing:latest",
        runner=_runner(returncode=1, stderr="no such manifest: brainles/missing"),
    )
    assert result.state == "not_found"
    assert not result.pullable


def test_manifest_inspect_private():
    result = manifest_inspect(
        "brainles/private:latest",
        runner=_runner(returncode=1, stderr="unauthorized: authentication required"),
    )
    assert result.state == "private"


def test_manifest_inspect_offline():
    result = manifest_inspect(
        "brainles/example:latest",
        runner=_runner(returncode=1, stderr="dial tcp: i/o timeout"),
    )
    assert result.state == "offline"


def test_layer_sizes_filters_to_amd64_verbose_payload():
    payload = json.dumps(
        [
            {
                "Descriptor": {"platform": {"architecture": "arm64", "os": "linux"}},
                "OCIManifest": {"layers": [{"size": 5_000_000_000}]},
            },
            {
                "Descriptor": {"platform": {"architecture": "amd64", "os": "linux"}},
                "OCIManifest": {"layers": [{"size": 1_000_000_000}]},
            },
        ]
    )
    result = manifest_inspect("brainles/example:latest", runner=_runner(stdout=payload))
    assert result.state == "pullable"
    assert result.size_gb == pytest.approx(1.0)
