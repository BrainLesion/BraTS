"""End-to-end smoke-test suite for the containerized BraTS algorithms.

This script runs a small, representative selection of algorithms through the
full inference workflow on de-identified sample data and validates the output
structurally. It is intentionally opt-in and resource-heavy: it needs a working
container backend (Docker by default), network access to fetch the sample data
and container images, and for most algorithms a GPU.

Usage:
    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --case inpainting --keep
    uv run python scripts/smoke_test.py --data /path/to/data --junit-xml report.xml

The same logic backs the opt-in ``pytest -m smoke`` tests, so the script can be
run directly on any server without going through pytest.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

import brats
from brats import constants
from brats.constants import Backends

DEFAULT_MANIFEST = Path(__file__).parent / "smoke_manifest.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned commit of BrainLesion/tutorials so downloaded sample data is stable.
DATA_REF = "99c5841fe8985e4004d3d5394ca9590a914e24e7"
DATA_BASE_URL = "https://raw.githubusercontent.com/BrainLesion/tutorials"
SUBJECT_ID = "BraTS-GLI-00001-000"

# Modalities contained in each sample dataset (file suffix after "<subject>-").
DATA_FILES: dict[str, list[str]] = {
    "segmentation": ["t1c", "t1n", "t2f", "t2w"],
    "inpainting": ["mask", "t1n-voided"],
}

# Keyword name used to call ``infer_single`` mapped to the file suffix on disk.
INPAINTING_INPUTS = {"t1n": "t1n-voided", "mask": "mask"}

DEFAULT_SEGMENTATION_LABELS = [0, 1, 2, 3, 4]

_DEFAULT_DATA_ROOT = Path.home() / ".cache" / "brats" / "smoke"


class SmokeError(RuntimeError):
    """Base class for smoke-test failures."""


class SmokeValidationError(SmokeError):
    """Raised when an algorithm output does not pass structural validation."""


@dataclass
class SmokeCase:
    """A single smoke-test case loaded from the manifest."""

    id: str
    task: str
    algorithm_class: str
    algorithm: str
    data: str
    modalities: Optional[list[str]] = None
    synthesize: Optional[str] = None
    kwargs: dict[str, object] = field(default_factory=dict)
    allowed_labels: Optional[list[int]] = None


@dataclass
class CaseResult:
    """Outcome of a single smoke-test case."""

    case_id: str
    status: str
    duration: float
    message: str = ""
    output_file: Optional[Path] = None


def load_manifest(path: Path) -> tuple[dict[str, object], list[SmokeCase]]:
    """Load the smoke-test manifest.

    Args:
        path (Path): Path to the YAML manifest

    Returns:
        Tuple[Dict[str, object], List[SmokeCase]]: Manifest defaults and cases
    """
    with open(path) as file:
        raw = yaml.safe_load(file) or {}

    defaults = raw.get("defaults") or {}
    cases = [
        SmokeCase(
            id=entry["id"],
            task=entry["task"],
            algorithm_class=entry["algorithm_class"],
            algorithm=entry["algorithm"],
            data=entry["data"],
            modalities=entry.get("modalities"),
            synthesize=entry.get("synthesize"),
            kwargs=entry.get("kwargs") or {},
            allowed_labels=entry.get("allowed_labels"),
        )
        for entry in raw.get("cases") or []
    ]
    if not cases:
        raise SmokeError(f"No smoke-test cases found in {path}")
    return defaults, cases


def select_cases(cases: list[SmokeCase], case_ids: list[str]) -> list[SmokeCase]:
    """Filter cases by identifier, raising on unknown identifiers."""
    known = {case.id for case in cases}
    unknown = set(case_ids) - known
    if unknown:
        raise SmokeError(
            f"Unknown smoke-test case(s): {sorted(unknown)}. Available: {sorted(known)}"
        )
    return [case for case in cases if case.id in set(case_ids)]


def resolve_data_root(cli_data: Optional[str]) -> Path:
    """Resolve the sample-data root from CLI, environment or default cache."""
    if cli_data:
        return Path(cli_data).expanduser()
    env_data = os.environ.get("BRATS_SMOKE_DATA")
    if env_data:
        return Path(env_data).expanduser()
    return _DEFAULT_DATA_ROOT


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _disk_check_path(data_root: Optional[Path]) -> Path:
    """Return the nearest existing ancestor of ``data_root`` (or the repo root)."""
    path = (data_root or REPO_ROOT).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_environment(
    backend: str,
    force_cpu: bool,
    min_free_gb: float = 5.0,
    data_root: Optional[Path] = None,
) -> list[str]:
    """Check host preconditions and return a list of problems.

    Args:
        backend (str): Container backend to use
        force_cpu (bool): Whether CPU execution was requested
        min_free_gb (float): Minimum required free disk space in GB
        data_root (Optional[Path]): Folder the sample data is cached in. The
            disk-space check runs against its nearest existing ancestor

    Returns:
        List[str]: Human-readable problems; empty if the host is ready
    """
    problems: list[str] = []

    if backend == Backends.DOCKER.value:
        if shutil.which("docker") is None:
            problems.append("docker CLI not found on PATH")
        else:
            try:
                proc = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if proc.returncode != 0:
                    problems.append("Docker daemon is not reachable")
            except (OSError, subprocess.SubprocessError) as e:
                problems.append(f"Failed to run docker: {e}")
    elif backend == Backends.SINGULARITY.value and (
        shutil.which("singularity") is None and shutil.which("apptainer") is None
    ):
        problems.append("singularity/apptainer not found on PATH")

    if not force_cpu:
        if shutil.which("nvidia-smi") is None:
            problems.append(
                "nvidia-smi not found; pass --force-cpu if the selected "
                "algorithms are CPU-compatible"
            )
        else:
            try:
                proc = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                if proc.returncode != 0:
                    problems.append("nvidia-smi reported no usable GPU")
            except (OSError, subprocess.SubprocessError) as e:
                problems.append(f"Failed to run nvidia-smi: {e}")

    try:
        free_gb = shutil.disk_usage(_disk_check_path(data_root)).free / (1024**3)
        if free_gb < min_free_gb:
            problems.append(
                f"Only {free_gb:.1f} GB free disk space "
                f"(at least {min_free_gb:.0f} GB recommended)"
            )
    except OSError as e:
        problems.append(f"Could not determine free disk space: {e}")

    return problems


def download_sample_data(dataset: str, data_root: Path, ref: str = DATA_REF) -> Path:
    """Ensure the sample dataset is available locally and return its path.

    Already present files are reused, so the download only happens once. The
    returned folder contains one subject in the standard BraTS layout.

    Args:
        dataset (str): Dataset name ("segmentation" or "inpainting")
        data_root (Path): Root folder for the cached sample data
        ref (str): Git ref of BrainLesion/tutorials to download from

    Returns:
        Path: Folder containing the subject files
    """
    if dataset not in DATA_FILES:
        raise SmokeError(f"Unknown sample dataset: {dataset}")

    subject_dir = data_root / "BraTS" / "data" / dataset / SUBJECT_ID
    subject_dir.mkdir(parents=True, exist_ok=True)

    for modality in DATA_FILES[dataset]:
        destination = subject_dir / f"{SUBJECT_ID}-{modality}.nii.gz"
        if destination.exists() and destination.stat().st_size > 0:
            continue
        url = (
            f"{DATA_BASE_URL}/{ref}/BraTS/data/{dataset}/"
            f"{SUBJECT_ID}/{SUBJECT_ID}-{modality}.nii.gz"
        )
        _download(url=url, destination=destination)

    return subject_dir


def _download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` atomically."""
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "brats-smoke-test"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: SIM117
            with open(tmp, "wb") as file:
                shutil.copyfileobj(response, file)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise SmokeError(f"Failed to download {url}: {e}") from e

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise SmokeError(f"Downloaded empty file from {url}")
    tmp.replace(destination)


def _resolve_algorithm_class(class_name: str):
    cls = getattr(brats, class_name, None)
    if cls is None:
        raise SmokeError(f"Unknown algorithm class exported from brats: {class_name}")
    return cls


def _resolve_algorithm(algorithm: str):
    enum_name, _, member_name = algorithm.partition(".")
    enum_cls = getattr(constants, enum_name, None)
    if enum_cls is None:
        raise SmokeError(f"Unknown algorithm enum: {enum_name}")
    try:
        return enum_cls[member_name]
    except KeyError as e:
        raise SmokeError(f"Unknown algorithm member: {algorithm}") from e


def build_inputs(case: SmokeCase, subject_dir: Path) -> dict[str, Path]:
    """Build the ``infer_single`` inputs for a case from the sample data."""
    if case.task == "inpainting":
        return {
            name: subject_dir / f"{SUBJECT_ID}-{suffix}.nii.gz"
            for name, suffix in INPAINTING_INPUTS.items()
        }

    modalities = case.modalities or ["t1c", "t1n", "t2f", "t2w"]
    return {
        modality: subject_dir / f"{SUBJECT_ID}-{modality}.nii.gz"
        for modality in modalities
    }


@contextmanager
def _time_limit(seconds: Optional[float]) -> Iterator[None]:
    """Raise ``TimeoutError`` if the wrapped block runs longer than ``seconds``."""
    if seconds is None or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Case exceeded the {seconds:.0f}s time limit")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _validate_output(
    output_file: Path,
    reference_file: Path,
    task: str,
    allowed_labels: Optional[list[int]],
    synthesize: Optional[str],
) -> None:
    """Validate an algorithm output structurally."""
    if not output_file.exists():
        raise SmokeValidationError(f"Output file was not created: {output_file}")
    if output_file.stat().st_size == 0:
        raise SmokeValidationError(f"Output file is empty: {output_file}")

    try:
        output_img = nib.load(str(output_file))
        data = np.asarray(output_img.dataobj)
    except Exception as e:
        raise SmokeValidationError(f"Output is not a readable NIfTI: {e}") from e

    if data.size == 0:
        raise SmokeValidationError("Output volume has zero voxels")
    if not np.isfinite(data).all():
        raise SmokeValidationError("Output contains non-finite values")
    if np.count_nonzero(data) == 0:
        raise SmokeValidationError("Output contains only zeros")

    reference_img = nib.load(str(reference_file))
    if output_img.shape != reference_img.shape:
        raise SmokeValidationError(
            f"Shape mismatch: got {output_img.shape}, expected {reference_img.shape}"
        )
    if not np.allclose(output_img.affine, reference_img.affine, atol=1e-3):
        raise SmokeValidationError("Output affine does not match the input affine")

    if task == "segmentation":
        values = np.unique(data)
        if not np.allclose(values, np.round(values), atol=1e-4):
            raise SmokeValidationError(
                f"Segmentation contains non-integer labels: {values[:5]}"
            )
        allowed = set(allowed_labels or DEFAULT_SEGMENTATION_LABELS)
        unexpected = {int(value) for value in values} - allowed
        if unexpected:
            raise SmokeValidationError(
                f"Segmentation contains unexpected label(s): {sorted(unexpected)}"
            )
    elif synthesize:
        # Missing-MRI output is a synthesized intensity image: only finiteness
        # and shape/affine are meaningful here.
        pass


def run_case(
    case: SmokeCase,
    *,
    data_root: Path,
    backend: str = Backends.DOCKER.value,
    force_cpu: bool = False,
    cuda_devices: str = "0",
    workspace: Path,
    timeout: Optional[float] = None,
) -> CaseResult:
    """Run a single smoke-test case end to end.

    Args:
        case (SmokeCase): Case to run
        data_root (Path): Root folder with (or for) the sample data
        backend (str): Container backend to use
        force_cpu (bool): Whether to force CPU execution
        cuda_devices (str): CUDA devices to expose
        workspace (Path): Folder for outputs and logs
        timeout (Optional[float]): Per-case timeout in seconds

    Returns:
        CaseResult: The outcome of the case
    """
    start = time.time()
    output_dir = workspace / case.id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.nii.gz"

    try:
        algorithm_class = _resolve_algorithm_class(case.algorithm_class)
        algorithm_member = _resolve_algorithm(case.algorithm)
        instance = algorithm_class(
            algorithm=algorithm_member,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
            **case.kwargs,
        )

        if force_cpu and not instance.algorithm.run_args.cpu_compatible:
            raise SmokeValidationError(
                f"Algorithm {case.algorithm} is not CPU-compatible and cannot "
                "run with --force-cpu"
            )

        subject_dir = download_sample_data(dataset=case.data, data_root=data_root)
        inputs = build_inputs(case=case, subject_dir=subject_dir)
        missing = [str(path) for path in inputs.values() if not path.exists()]
        if missing:
            raise SmokeValidationError(f"Missing input files: {missing}")

        output_file.unlink(missing_ok=True)
        log_file = output_dir / "inference.log"

        with _time_limit(timeout):
            instance.infer_single(
                output_file=output_file,
                log_file=log_file,
                backend=Backends(backend),
                **inputs,
            )

        reference_file = next(iter(inputs.values()))
        _validate_output(
            output_file=output_file,
            reference_file=reference_file,
            task=case.task,
            allowed_labels=case.allowed_labels,
            synthesize=case.synthesize,
        )
        return CaseResult(
            case_id=case.id,
            status="passed",
            duration=time.time() - start,
            output_file=output_file,
        )
    except TimeoutError as e:
        return CaseResult(
            case_id=case.id,
            status="failed",
            duration=time.time() - start,
            message=str(e),
            output_file=output_file,
        )
    except Exception as e:  # noqa: BLE001 (report any failure per case)
        return CaseResult(
            case_id=case.id,
            status="failed",
            duration=time.time() - start,
            message=f"{type(e).__name__}: {e}",
            output_file=output_file,
        )


def write_junit_xml(results: list[CaseResult], path: Path) -> None:
    """Write a JUnit XML report for the given results."""
    failures = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    suite = ET.Element(
        "testsuite",
        name="brats-smoke",
        tests=str(len(results)),
        failures=str(failures),
        skipped=str(skipped),
        errors="0",
        time=f"{sum(r.duration for r in results):.3f}",
    )
    for result in results:
        testcase = ET.SubElement(
            suite,
            "testcase",
            classname="brats.smoke",
            name=result.case_id,
            time=f"{result.duration:.3f}",
        )
        if result.status == "failed":
            ET.SubElement(testcase, "failure", message=result.message)
        elif result.status == "skipped":
            ET.SubElement(testcase, "skipped", message=result.message)

    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _print_report(results: list[CaseResult], console: Console) -> None:
    table = Table(title="BraTS smoke tests", show_lines=True)
    table.add_column("case", style="cyan", no_wrap=True)
    table.add_column("status", justify="center")
    table.add_column("time", justify="right")
    table.add_column("details", overflow="fold")

    status_style = {
        "passed": "[green]passed[/green]",
        "failed": "[red]failed[/red]",
        "skipped": "[yellow]skipped[/yellow]",
    }
    for result in results:
        table.add_row(
            result.case_id,
            status_style.get(result.status, result.status),
            f"{result.duration:.1f}s",
            result.message or "-",
        )
    console.print(table)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run end-to-end smoke tests for containerized BraTS algorithms."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the smoke-test selection manifest",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        default=None,
        help="Run only the given case id (repeatable)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Root folder for the sample data "
        "(default: $BRATS_SMOKE_DATA or ~/.cache/brats/smoke)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=[b.value for b in Backends],
        help="Container backend to use (default: manifest default)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        default=None,
        help="Force CPU execution (only for CPU-compatible algorithms)",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help='Comma-separated CUDA device ids, e.g. "0,1"',
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-case timeout in seconds (default: manifest 'timeout', or no limit)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep output and log files for debugging",
    )
    parser.add_argument(
        "--junit-xml",
        default=None,
        help="Write a JUnit XML report to this path",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    console = Console()

    try:
        defaults, cases = load_manifest(Path(args.manifest))
        if args.cases:
            cases = select_cases(cases, args.cases)
    except SmokeError as e:
        console.print(f"[red]Manifest error:[/red] {e}")
        return 2

    backend = (
        args.backend
        or os.environ.get("BRATS_SMOKE_BACKEND")
        or str(defaults.get("backend", Backends.DOCKER.value))
    )
    force_cpu = bool(
        args.force_cpu
        if args.force_cpu is not None
        else defaults.get("force_cpu", False)
    ) or env_truthy("BRATS_SMOKE_FORCE_CPU")
    cuda_devices = (
        args.cuda_devices
        or os.environ.get("BRATS_SMOKE_CUDA_DEVICES")
        or str(defaults.get("cuda_devices", "0"))
    )
    data_root = resolve_data_root(args.data)
    timeout_env = os.environ.get("BRATS_SMOKE_TIMEOUT")
    if args.timeout is not None:
        timeout = args.timeout
    elif timeout_env is not None:
        timeout = float(timeout_env)
    else:
        timeout = defaults.get("timeout")

    problems = check_environment(
        backend=backend, force_cpu=force_cpu, data_root=data_root
    )
    if problems:
        console.print("[red]Environment is not ready:[/red]")
        for problem in problems:
            console.print(f"  - {problem}")
        return 2

    console.print(
        f"Running {len(cases)} smoke-test case(s) with backend "
        f"[bold]{backend}[/bold]"
        + (" (CPU forced)" if force_cpu else "")
        + f"\nSample data root: {data_root}"
    )

    keep = args.keep or env_truthy("BRATS_SMOKE_KEEP")
    workspace = Path(tempfile.mkdtemp(prefix="brats_smoke_"))
    results = [
        run_case(
            case,
            data_root=data_root,
            backend=backend,
            force_cpu=force_cpu,
            cuda_devices=cuda_devices,
            workspace=workspace,
            timeout=timeout,
        )
        for case in cases
    ]
    if keep:
        console.print(f"[yellow]Keeping workspace: {workspace}[/yellow]")

    _print_report(results, console)

    if args.junit_xml:
        write_junit_xml(results, Path(args.junit_xml))
        console.print(f"Wrote JUnit XML report to {args.junit_xml}")

    if not keep:
        shutil.rmtree(workspace, ignore_errors=True)

    failed = [result for result in results if result.status == "failed"]
    if failed:
        console.print(f"[red]{len(failed)} case(s) failed.[/red]")
        return 1
    console.print("[green]All smoke-test cases passed.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
