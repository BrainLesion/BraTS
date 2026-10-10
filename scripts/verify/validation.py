"""Structural output validation for the verification suite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from scripts._shared import DEFAULT_SEGMENTATION_LABELS, validate_output
from scripts.verify.datasets import DatasetError, spec_for
from scripts.verify.models import AlgorithmCase

# Allowed label overrides for outputs whose label space differs from the default
# BraTS {0, 1, 2, 3, 4}. Keys are track slugs (``case.slug``) or, for
# per-algorithm exceptions, exact case ids. An explicit entry makes the label
# check strict; without one, unexpected labels are only warned about (see
# ``validate_case_output``).
_ALLOWED_LABELS_BY_TRACK: dict[str, list[int]] = {}
_ALLOWED_LABELS_BY_CASE: dict[str, list[int]] = {}


def allowed_labels_for(case: AlgorithmCase) -> Optional[list[int]]:
    """Return the allowed label set for a case, if explicitly configured."""
    if case.case_id in _ALLOWED_LABELS_BY_CASE:
        return _ALLOWED_LABELS_BY_CASE[case.case_id]
    return _ALLOWED_LABELS_BY_TRACK.get(case.slug)


def validate_case_output(
    case: AlgorithmCase,
    output_file: Path,
    reference_file: Path,
) -> None:
    """Validate a case's output, raising ``ValidationError`` on failure.

    Args:
        case (AlgorithmCase): The case that produced the output
        output_file (Path): The output file to validate
        reference_file (Path): Input file for shape/affine comparison

    Raises:
        scripts._shared.ValidationError: If the output is structurally invalid
    """
    task = case.task
    if task == "segmentation":
        override = allowed_labels_for(case)
        allowed_labels = (
            override if override is not None else DEFAULT_SEGMENTATION_LABELS
        )
        # Only enforce the label set when an algorithm-specific override exists.
        # Otherwise report unfamiliar labels as a warning instead of failing the
        # case, since we do not know every algorithm's label convention.
        strict_labels = override is not None
        synthesize = None
    else:
        allowed_labels = None
        strict_labels = True
        try:
            synthesize = spec_for(case).synthesize
        except DatasetError:
            synthesize = None

    def _warn(message: str) -> None:
        logger.warning(f"{case.case_id}: {message}")

    validate_output(
        output_file=output_file,
        reference_file=reference_file,
        task=task,
        allowed_labels=allowed_labels,
        synthesize=synthesize,
        strict_labels=strict_labels,
        warn=_warn,
    )
