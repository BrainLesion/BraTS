"""Fixtures and reporting hooks for the integration verification suite.

The heavy, container/GPU-backed cases are excluded from the default test run
(see ``addopts`` in ``pyproject.toml``). Run them with::

    uv run pytest tests/integration -m integration --data ~/brats_data
    uv run pytest tests/integration -m smoke --data ~/brats_data
    uv run pytest tests/integration -m network      # registry reachability only

The GPU gate is applied per case inside ``run_case``; this module only wires up
the environment check, the shared workspace and the result reporting.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from loguru import logger

from brats.utils import logging as brats_logging
from scripts._shared import check_environment
from scripts.verify.models import CaseResult
from scripts.verify.reporting import render_summary, write_json
from scripts.verify.runner import RunOptions, cleanup_owned_images

_DEFAULT_DATA_ROOT = Path.home() / "brats_data"
_DEFAULT_REPORT = Path("reports") / "verify-results.json"

_RECORDS: list[CaseResult] = []


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("brats-verify")
    group.addoption(
        "--data",
        default=os.environ.get("BRATS_VERIFY_DATA", str(_DEFAULT_DATA_ROOT)),
        help="Root folder containing the challenge data",
    )
    group.addoption(
        "--cuda-devices",
        default=os.environ.get("BRATS_VERIFY_CUDA_DEVICES", "0"),
        help='Comma-separated CUDA device ids, e.g. "1"',
    )
    group.addoption(
        "--force-cpu",
        action="store_true",
        help="Skip the GPU gate (CPU-compatible algorithms only)",
    )
    group.addoption(
        "--timeout", type=float, default=3600.0, help="Per-case timeout in seconds"
    )
    group.addoption(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("BRATS_VERIFY_MIN_FREE_GB", "30")),
        help="Minimum free disk space floor in GB (per filesystem)",
    )
    group.addoption(
        "--disk-path",
        default=None,
        help="Extra path whose filesystem must have free space",
    )
    group.addoption(
        "--workspace",
        default=os.environ.get("BRATS_VERIFY_WORKSPACE"),
        help="Folder for per-case outputs/weights (default: a temp dir)",
    )
    group.addoption(
        "--use-cached-weights",
        action="store_true",
        help="Use the shared Zenodo cache instead of downloading fresh weights "
        "(the shared cache is never deleted)",
    )
    group.addoption(
        "--gpu-idle-mem-mb",
        type=int,
        default=500,
        help="Maximum tolerated GPU memory usage in MB",
    )
    group.addoption(
        "--gpu-wait-seconds",
        type=int,
        default=600,
        help="Maximum seconds to wait for the GPU to free up",
    )
    group.addoption(
        "--gpu-poll-seconds",
        type=int,
        default=20,
        help="GPU polling interval in seconds",
    )
    group.addoption(
        "--keep", action="store_true", help="Keep the per-case workspace and logs"
    )
    group.addoption(
        "--verify-report",
        default=os.environ.get("BRATS_VERIFY_REPORT", str(_DEFAULT_REPORT)),
        help="Path for the JSON verification report",
    )


def pytest_configure(config: pytest.Config) -> None:
    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses not in (None, 0, "0"):
        raise pytest.UsageError(
            "The verification suite must run single-process: the GPU-free gate "
            "assumes no concurrent cases. Do not use pytest-xdist (-n)."
        )


@pytest.fixture(scope="session")
def verify_data_root(request: pytest.FixtureRequest) -> Path:
    """Return the root folder containing the challenge data."""
    return Path(request.config.getoption("--data")).expanduser()


@pytest.fixture(scope="session")
def verify_options(
    request: pytest.FixtureRequest, verify_data_root: Path
) -> RunOptions:
    """Build run options once and check the host environment."""
    # `import brats` calls `logger.disable("brats")`, so the per-case log files
    # written by `InferenceSetup` would stay empty. Re-enable the package logging
    # (and keep the default stderr sink quiet) so failures stay debuggable.
    logger.remove()
    brats_logging.add_console_handler("WARNING")
    brats_logging.enable()

    force_cpu = bool(request.config.getoption("--force-cpu"))
    problems = check_environment(
        backend="docker",
        force_cpu=force_cpu,
        min_free_gb=0.0,
        data_root=verify_data_root,
    )
    if problems:
        pytest.skip("Environment is not ready: " + "; ".join(problems))

    timeout = float(request.config.getoption("--timeout"))
    use_cached_weights = bool(request.config.getoption("--use-cached-weights"))
    keep = bool(request.config.getoption("--keep"))
    workspace_opt = request.config.getoption("--workspace")
    if workspace_opt:
        base_workspace = Path(workspace_opt).expanduser()
        generated_root = False
    else:
        base_workspace = Path(tempfile.mkdtemp(prefix="brats_verify_"))
        generated_root = True
    # Always create a session-unique subdirectory: teardown removes the
    # workspace, so reusing a user-provided root must never delete artifacts
    # left behind by an earlier run.
    workspace = base_workspace / f"session-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=True)

    disk_path_opt = request.config.getoption("--disk-path")
    options = RunOptions(
        data_root=verify_data_root,
        reports_dir=Path("reports") / "verify",
        workspace=workspace,
        cuda_devices=str(request.config.getoption("--cuda-devices")),
        force_cpu=force_cpu,
        timeout=timeout if timeout > 0 else None,
        min_free_gb=float(request.config.getoption("--min-free-gb")),
        disk_path=Path(disk_path_opt).expanduser() if disk_path_opt else None,
        gpu_idle_mem_mb=int(request.config.getoption("--gpu-idle-mem-mb")),
        gpu_wait_seconds=int(request.config.getoption("--gpu-wait-seconds")),
        gpu_poll_seconds=int(request.config.getoption("--gpu-poll-seconds")),
        # A shared cache is never deleted, so keep_weights follows use_cached_weights.
        fresh_weights=not use_cached_weights,
        keep_weights=use_cached_weights,
        keep=keep,
    )
    yield options

    cleanup_owned_images(options)
    if not keep:
        shutil.rmtree(options.workspace, ignore_errors=True)
        if generated_root:
            shutil.rmtree(base_workspace, ignore_errors=True)


@pytest.fixture(scope="session")
def verify_records() -> list[CaseResult]:
    """Shared collector for per-case results (used by the report hooks)."""
    return _RECORDS


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    if not _RECORDS:
        return
    render_summary(terminalreporter.write_line, _RECORDS)
    report = write_json(_RECORDS, Path(config.getoption("--verify-report")))
    terminalreporter.write_line("")
    terminalreporter.write_line(f"JSON report written to {report}")
