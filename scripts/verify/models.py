"""Data models for the all-algorithms verification suite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from brats.constants import Algorithms


class Status(str, Enum):
    """Terminal status of a single verification case."""

    OK = "ok"
    EXPECTED_FAIL = "expected_fail"
    PREFLIGHT_FAIL = "preflight_fail"
    PULL_FAIL = "pull_fail"
    DISK_SKIPPED = "disk_skipped"
    GPU_BUSY_SKIPPED = "gpu_busy_skipped"
    RUN_FAIL = "run_fail"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    ENV_ERROR = "env_error"

    @property
    def is_failure(self) -> bool:
        """Whether this status counts as a failure (as opposed to a skip)."""
        return self in _FAILURE_STATUSES

    @property
    def is_skipped(self) -> bool:
        """Whether this status counts as a skip."""
        return self in _SKIPPED_STATUSES


_FAILURE_STATUSES = frozenset(
    {
        Status.PREFLIGHT_FAIL,
        Status.PULL_FAIL,
        Status.RUN_FAIL,
        Status.INVALID_OUTPUT,
        Status.TIMEOUT,
        Status.ENV_ERROR,
    }
)

_SKIPPED_STATUSES = frozenset({Status.DISK_SKIPPED, Status.GPU_BUSY_SKIPPED})


@dataclass(frozen=True)
class AlgorithmCase:
    """A single algorithm invocation to verify."""

    track: str
    """Enum class name, e.g. ``"InpaintingAlgorithms"``."""
    slug: str
    """Short track identifier used by the CLI and dataset registry, e.g.
    ``"inpainting"``."""
    member: str
    """Algorithm enum member, e.g. ``"BraTS23_1"``."""
    case_id: str
    """Unique identifier, e.g. ``"inpainting.BraTS23_1"`` or
    ``"adult_glioma_pre_post.BraTS25_1:post"``."""
    class_name: str
    """Public algorithm class exported from :mod:`brats`, e.g. ``"Inpainter"``."""
    algorithm: Algorithms
    """Selected algorithm enum member."""
    docker_image: str
    """Container image reference."""
    year: int
    """Challenge year the algorithm was submitted to."""
    task: str
    """One of ``segmentation``, ``inpainting`` or ``missing_mri``."""
    dataset_key: str
    """Key into the dataset registry."""
    kwargs: dict[str, object] = field(default_factory=dict)
    """Extra constructor keyword arguments (e.g. ``treatment_timepoint``)."""
    requires_weights: bool = False
    """Whether the algorithm downloads Zenodo additional files."""
    record_id: Optional[str] = None
    """Zenodo record id of the additional files, if any."""
    parameters_file: bool = False
    """Whether the algorithm requires a parameters file."""
    treatment_timepoint: Optional[str] = None
    """Treatment timepoint variant (``pre``/``post``), if applicable."""
    known_failure_reason: Optional[str] = None
    """Reason this case is expected to fail, if registered."""


@dataclass
class CaseResult:
    """Outcome of a single verification case."""

    case: AlgorithmCase
    status: Status
    duration_s: float
    message: str = ""
    output_file: Optional[Path] = None
    log_file: Optional[Path] = None
    image_pulled_by_us: bool = False
    image_id: Optional[str] = None
