"""Build verification cases by introspecting the :mod:`brats` registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import yaml

from brats.constants import Algorithms
from brats.core.brats_algorithm import BraTSAlgorithm
from brats.registry import AlgorithmRecord, iter_algorithms
from scripts.verify.datasets import _SPECS
from scripts.verify.models import AlgorithmCase

DEFAULT_KNOWN_FAILURES = Path(__file__).resolve().parent.parent / "known_failures.yaml"

# Metadata file stems that differ from the dataset-registry slug.
_SLUG_ALIASES: dict[str, str] = {
    "adult_glioma_pre_and_post_treatment": "adult_glioma_pre_post",
}

# Curated subset for a quick end-to-end check against server data. Covers the
# main algorithm (Adult Glioma Pre&Post), the newest years (2025/2026) and one
# lightweight Zenodo-weighted case exercising the additional-files download path.
# Cases are kept fast: the slow `inpainting.BraTS24_3` (smint) is intentionally
# not used here.
SMOKE_CASE_IDS: frozenset[str] = frozenset(
    {
        "adult_glioma_pre_post.BraTS25_3B:pre",
        "adult_glioma_pre_post.BraTS25_3B:post",
        "inpainting.BraTS26_2B",
        "missing_mri.BraTS25_3",
        "missing_mri.BraTS24_2",
    }
)


class DiscoveryError(RuntimeError):
    """Raised when the algorithm registry cannot be introspected."""


def load_known_failures(path: Path = DEFAULT_KNOWN_FAILURES) -> dict[str, str]:
    """Load the image -> reason known-failure registry (empty if absent)."""
    if not path.exists():
        return {}
    with open(path) as file:
        return yaml.safe_load(file) or {}


def slug_for(record: AlgorithmRecord) -> str:
    """Return the dataset-registry slug for an algorithm record."""
    return _SLUG_ALIASES.get(record.track, record.track)


def _public_classes_by_enum() -> dict[type[Algorithms], type[BraTSAlgorithm]]:
    """Map each public enum class to the algorithm class that uses it."""
    brats = importlib.import_module("brats")
    mapping: dict[type[Algorithms], type[BraTSAlgorithm]] = {}
    for obj in vars(brats).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BraTSAlgorithm)
            and obj is not BraTSAlgorithm
        ):
            mapping[obj.algorithm_enum] = obj
    return mapping


def _timepoints(record: AlgorithmRecord) -> list[Optional[str]]:
    """Return the timepoint variants to verify for an algorithm.

    Tracks that encode a treatment timepoint advertise it through
    ``run_args.suffix_by_timepoint``; all others are verified once.
    """
    suffix_by_timepoint = record.data.run_args.suffix_by_timepoint
    if not suffix_by_timepoint:
        return [None]
    return list(suffix_by_timepoint)


def build_cases(
    known_failures: Optional[dict[str, str]] = None,
) -> list[AlgorithmCase]:
    """Build the full list of verification cases.

    Args:
        known_failures (Optional[Dict[str, str]]): Image -> reason mapping for
            algorithms that are expected to fail. Loaded from the default
            registry when omitted.

    Returns:
        List[AlgorithmCase]: One or more cases per registered algorithm

    Raises:
        DiscoveryError: If an algorithm has no matching public class or the
            track has no dataset mapping
    """
    failures = known_failures if known_failures is not None else load_known_failures()
    classes = _public_classes_by_enum()

    cases: list[AlgorithmCase] = []
    for record in iter_algorithms():
        cls = classes.get(record.enum_cls)
        if cls is None:
            raise DiscoveryError(
                f"No public algorithm class declares algorithm_enum "
                f"{record.enum_cls.__name__}"
            )
        slug = slug_for(record)
        if slug not in _SPECS:
            raise DiscoveryError(
                f"No dataset mapping registered for {record.enum_cls.__name__} "
                f"(slug {slug!r})"
            )
        spec = _SPECS[slug]
        additional = record.data.additional_files
        for timepoint in _timepoints(record):
            kwargs: dict[str, object] = {}
            member = record.member
            case_id = f"{slug}.{record.member}"
            if timepoint is not None:
                kwargs["treatment_timepoint"] = timepoint
                member = f"{record.member}:{timepoint}"
                case_id = f"{case_id}:{timepoint}"
            cases.append(
                AlgorithmCase(
                    track=record.enum_cls.__name__,
                    slug=slug,
                    member=member,
                    case_id=case_id,
                    class_name=cls.__name__,
                    algorithm=record.algorithm,
                    docker_image=record.data.run_args.docker_image,
                    year=record.data.meta.year,
                    task=spec.task,
                    dataset_key=slug,
                    kwargs=kwargs,
                    requires_weights=additional is not None,
                    record_id=additional.record_id if additional is not None else None,
                    parameters_file=record.data.run_args.parameters_file,
                    treatment_timepoint=timepoint,
                    known_failure_reason=failures.get(
                        record.data.run_args.docker_image
                    ),
                )
            )
    return cases
