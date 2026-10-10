"""Track -> dataset mapping for the all-algorithms verification suite.

The plan's data mapping is verified against ``~/brats_data`` on the verification
host. Only the per-modality *file paths* matter: ``infer_single`` reassigns
internal names based on the algorithm metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scripts.verify.models import AlgorithmCase


class DatasetError(RuntimeError):
    """Raised when input data for a case cannot be resolved."""


# File name templates for the common 4-modality segmentation layout. The
# ``{subject}`` placeholder is replaced with the subject folder name.
_FOUR_MOD: dict[str, str] = {
    "t1c": "{subject}-t1c.nii.gz",
    "t1n": "{subject}-t1n.nii.gz",
    "t2f": "{subject}-t2f.nii.gz",
    "t2w": "{subject}-t2w.nii.gz",
}
_THREE_MOD: dict[str, str] = {
    "t1c": "{subject}-t1c.nii.gz",
    "t1n": "{subject}-t1n.nii.gz",
    "t2f": "{subject}-t2f.nii.gz",
}
_T1C: dict[str, str] = {"t1c": "{subject}-t1c.nii.gz"}
_INPAINTING: dict[str, str] = {
    "t1n": "{subject}-t1n-voided.nii.gz",
    "mask": "{subject}-mask.nii.gz",
}


@dataclass(frozen=True)
class DatasetSpec:
    """How to locate the inputs for a challenge track."""

    key: str
    task: str
    subject_dir: str
    filenames: dict[str, str]
    reference_key: str = "t1c"
    synthesize: Optional[str] = None
    subject_dir_by_timepoint: dict[str, str] = field(default_factory=dict)
    subject_dir_by_year: dict[int, str] = field(default_factory=dict)


_SPECS: dict[str, DatasetSpec] = {
    "adult_glioma_pre_treatment": DatasetSpec(
        key="adult_glioma_pre_treatment",
        task="segmentation",
        subject_dir="adult_glioma/BraTS-GLI-00001-000",
        filenames=_FOUR_MOD,
    ),
    "adult_glioma_pre_post": DatasetSpec(
        key="adult_glioma_pre_post",
        task="segmentation",
        subject_dir="adult_glioma_postop/BraTS-GLI-02073-100",
        filenames=_FOUR_MOD,
        subject_dir_by_timepoint={
            "pre": "adult_glioma/BraTS-GLI-00001-000",
            "post": "adult_glioma_postop/BraTS-GLI-02073-100",
        },
    ),
    "meningioma": DatasetSpec(
        key="meningioma",
        task="segmentation",
        subject_dir="adult_meningioma/BraTS-MEN-00000-000",
        filenames=_FOUR_MOD,
    ),
    "meningioma_rt": DatasetSpec(
        key="meningioma_rt",
        task="segmentation",
        subject_dir="adult_meningioma_24/BraTS-MEN-RT-0010-1",
        filenames={"t1c": "{subject}_t1c.nii.gz"},
    ),
    "pediatric": DatasetSpec(
        key="pediatric",
        task="segmentation",
        subject_dir="pediatric/BraTS-PED-00030-000",
        filenames=_FOUR_MOD,
    ),
    "africa": DatasetSpec(
        key="africa",
        task="segmentation",
        subject_dir="africa/BraTS-SSA-00126-000",
        filenames=_FOUR_MOD,
    ),
    "metastases": DatasetSpec(
        key="metastases",
        task="segmentation",
        subject_dir="metastases/BraTS-MET-00013-000",
        filenames=_FOUR_MOD,
        subject_dir_by_year={
            2023: "metastases/BraTS-MET-00013-000",
            2025: "MET25/BraTS-MET-00833-000",
        },
    ),
    "goat": DatasetSpec(
        key="goat",
        task="segmentation",
        subject_dir="adult_glioma/BraTS-GLI-00001-000",
        filenames=_FOUR_MOD,
    ),
    "inpainting": DatasetSpec(
        key="inpainting",
        task="inpainting",
        subject_dir="inpainting/BraTS-GLI-00001-000",
        filenames=_INPAINTING,
        reference_key="t1n",
    ),
    "missing_mri": DatasetSpec(
        key="missing_mri",
        task="missing_mri",
        subject_dir="missing_mri/BraTS-GLI-00001-000",
        filenames=_THREE_MOD,
        synthesize="t2w",
    ),
}


def track_slugs() -> list[str]:
    """Return the available track slugs."""
    return sorted(_SPECS)


def spec_for(case: AlgorithmCase) -> DatasetSpec:
    """Return the dataset spec for a case, raising on unknown keys."""
    try:
        return _SPECS[case.dataset_key]
    except KeyError as e:
        raise DatasetError(f"Unknown dataset key: {case.dataset_key}") from e


def resolve_subject_dir(case: AlgorithmCase, data_root: Path) -> Path:
    """Resolve the subject folder for a case."""
    spec = spec_for(case)
    relative = spec.subject_dir
    if case.treatment_timepoint and spec.subject_dir_by_timepoint:
        try:
            relative = spec.subject_dir_by_timepoint[case.treatment_timepoint]
        except KeyError as e:
            raise DatasetError(
                f"No subject dir for timepoint {case.treatment_timepoint!r}"
            ) from e
    elif spec.subject_dir_by_year:
        years = spec.subject_dir_by_year
        if case.year in years:
            relative = years[case.year]
        else:
            nearest = max((y for y in years if y <= case.year), default=min(years))
            relative = years[nearest]
    return data_root / relative


def resolve_inputs(case: AlgorithmCase, data_root: Path) -> dict[str, Path]:
    """Resolve the ``infer_single`` inputs for a case.

    Args:
        case (AlgorithmCase): The case to resolve inputs for
        data_root (Path): Root folder containing the challenge data

    Returns:
        Dict[str, Path]: Mapping of ``infer_single`` keyword to input file path

    Raises:
        DatasetError: If the dataset key is unknown or no subject dir matches
    """
    spec = spec_for(case)
    subject_dir = resolve_subject_dir(case, data_root)
    subject = subject_dir.name
    return {
        keyword: subject_dir / template.format(subject=subject)
        for keyword, template in spec.filenames.items()
    }


def resolve_reference(case: AlgorithmCase, inputs: dict[str, Path]) -> Path:
    """Return the input file to use as the shape/affine reference for validation."""
    spec = spec_for(case)
    if spec.reference_key in inputs:
        return inputs[spec.reference_key]
    return next(iter(inputs.values()))


def missing_inputs(inputs: dict[str, Path]) -> list[Path]:
    """Return the input paths that do not exist on disk."""
    return [path for path in inputs.values() if not path.exists()]
