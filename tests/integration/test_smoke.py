"""Opt-in end-to-end smoke tests (``pytest -m smoke``).

These tests are excluded from the default test run (see ``addopts`` in
``pyproject.toml``). They execute real containerized algorithms end to end and
therefore require a working container backend, network access and, for most
algorithms, a GPU. The logic lives in ``scripts/smoke_test.py`` so it can also
be run directly on any server.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from scripts import smoke_test

pytestmark = pytest.mark.smoke

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_manifest.yaml"
_DEFAULTS, CASES = smoke_test.load_manifest(_MANIFEST_PATH)


@pytest.fixture(scope="session")
def smoke_options() -> dict[str, object]:
    backend = os.environ.get(
        "BRATS_SMOKE_BACKEND", str(_DEFAULTS.get("backend", "docker"))
    )
    force_cpu = bool(_DEFAULTS.get("force_cpu", False)) or smoke_test.env_truthy(
        "BRATS_SMOKE_FORCE_CPU"
    )
    cuda_devices = os.environ.get(
        "BRATS_SMOKE_CUDA_DEVICES", str(_DEFAULTS.get("cuda_devices", "0"))
    )
    data_root = smoke_test.resolve_data_root(os.environ.get("BRATS_SMOKE_DATA"))
    timeout_env = os.environ.get("BRATS_SMOKE_TIMEOUT")
    timeout = (
        float(timeout_env) if timeout_env is not None else _DEFAULTS.get("timeout")
    )

    problems = smoke_test.check_environment(
        backend=backend, force_cpu=force_cpu, data_root=data_root
    )
    if problems:
        pytest.skip("Smoke-test environment not ready: " + "; ".join(problems))

    return {
        "backend": backend,
        "force_cpu": force_cpu,
        "cuda_devices": cuda_devices,
        "data_root": data_root,
        "timeout": float(timeout) if timeout else None,
    }


@pytest.fixture(scope="session")
def smoke_workspace() -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="brats_smoke_pytest_"))
    yield workspace
    if not smoke_test.env_truthy("BRATS_SMOKE_KEEP"):
        import shutil

        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_smoke_case(
    case: smoke_test.SmokeCase,
    smoke_options: dict[str, object],
    smoke_workspace: Path,
) -> None:
    result = smoke_test.run_case(
        case,
        data_root=smoke_options["data_root"],  # type: ignore[arg-type]
        backend=smoke_options["backend"],  # type: ignore[arg-type]
        force_cpu=smoke_options["force_cpu"],  # type: ignore[arg-type]
        cuda_devices=smoke_options["cuda_devices"],  # type: ignore[arg-type]
        workspace=smoke_workspace,
        timeout=smoke_options["timeout"],  # type: ignore[arg-type]
    )
    assert result.status == "passed", result.message
