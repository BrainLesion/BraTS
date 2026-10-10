"""Helpers shared by the verification suite's scripts and tests.

These utilities are intentionally free of any algorithm-specific knowledge so
that the host precondition checks, the timeout guard and the structural output
validation can be reused across the tooling.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np

from brats.constants import Backends

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SEGMENTATION_LABELS = [0, 1, 2, 3, 4]


class ValidationError(RuntimeError):
    """Raised when an algorithm output does not pass structural validation."""


def env_truthy(name: str) -> bool:
    """Return True if the environment variable ``name`` is set to a truthy value."""
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
        free_gb = shutil.disk_usage(_disk_check_path(data_root)).free / (1000**3)
        if free_gb < min_free_gb:
            problems.append(
                f"Only {free_gb:.1f} GB free disk space "
                f"(at least {min_free_gb:.0f} GB recommended)"
            )
    except OSError as e:
        problems.append(f"Could not determine free disk space: {e}")

    return problems


@contextmanager
def time_limit(seconds: Optional[float]) -> Iterator[None]:
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


def validate_output(
    output_file: Path,
    reference_file: Path,
    task: str,
    allowed_labels: Optional[list[int]] = None,
    synthesize: Optional[str] = None,
    default_labels: Optional[list[int]] = None,
    strict_labels: bool = True,
    warn: Optional[Callable[[str], None]] = None,
) -> None:
    """Validate an algorithm output structurally.

    Args:
        output_file (Path): The output file produced by the algorithm
        reference_file (Path): An input file used as shape/affine reference
        task (str): Task identifier (``segmentation``, ``inpainting`` or
            ``missing_mri``)
        allowed_labels (Optional[List[int]]): Allowed label set for
            segmentation outputs
        synthesize (Optional[str]): Modality synthesized by a missing-MRI
            algorithm (intensity output); only finiteness and shape are checked
        default_labels (Optional[List[int]]): Fallback label set when
            ``allowed_labels`` is not given
        strict_labels (bool): When True, unexpected segmentation labels raise
            ``ValidationError``. When False, they are reported through ``warn``
            instead (used when no algorithm-specific label set is known)
        warn (Optional[Callable[[str], None]]): Sink for non-fatal findings

    Raises:
        ValidationError: If the output fails any structural check
    """
    if not output_file.exists():
        raise ValidationError(f"Output file was not created: {output_file}")
    if output_file.stat().st_size == 0:
        raise ValidationError(f"Output file is empty: {output_file}")

    try:
        output_img = nib.load(str(output_file))
        data = np.asarray(output_img.dataobj)
    except Exception as e:
        raise ValidationError(f"Output is not a readable NIfTI: {e}") from e

    if data.size == 0:
        raise ValidationError("Output volume has zero voxels")
    if not np.isfinite(data).all():
        raise ValidationError("Output contains non-finite values")
    if np.count_nonzero(data) == 0:
        raise ValidationError("Output contains only zeros")

    reference_img = nib.load(str(reference_file))
    if output_img.shape != reference_img.shape:
        raise ValidationError(
            f"Shape mismatch: got {output_img.shape}, expected {reference_img.shape}"
        )
    if not np.allclose(output_img.affine, reference_img.affine, atol=1e-3):
        raise ValidationError("Output affine does not match the input affine")

    if task == "segmentation":
        values = np.unique(data)
        if not np.allclose(values, np.round(values), atol=1e-4):
            raise ValidationError(
                f"Segmentation contains non-integer labels: {values[:5]}"
            )
        allowed = set(
            allowed_labels
            if allowed_labels is not None
            else (default_labels or DEFAULT_SEGMENTATION_LABELS)
        )
        unexpected = {int(value) for value in values} - allowed
        if unexpected:
            message = f"Segmentation contains unexpected label(s): {sorted(unexpected)}"
            if strict_labels:
                raise ValidationError(message)
            if warn is not None:
                warn(message)
    elif synthesize:
        # Missing-MRI output is a synthesized intensity image: only finiteness
        # and shape/affine are meaningful here.
        pass
