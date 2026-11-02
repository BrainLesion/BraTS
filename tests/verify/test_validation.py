"""Unit tests for structural output validation."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts._shared import ValidationError
from scripts.verify import validation
from scripts.verify.cases import build_cases
from scripts.verify.validation import validate_case_output


def _write(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), str(path))


def _segmentation_case():
    return next(
        case
        for case in build_cases()
        if case.slug == "pediatric" and case.member == "BraTS25_1"
    )


def test_valid_segmentation(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    _write(reference, reference_data)
    _write(output, np.array(reference_data, dtype=np.int16))
    validate_case_output(case, output, reference)


def test_non_integer_labels_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((3, 3, 3), dtype=np.float32)
    _write(reference, reference_data)
    bad = reference_data.copy()
    bad[0, 0, 0] = 1.5
    _write(output, bad)
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def test_shape_mismatch_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    _write(reference, np.ones((4, 4, 4), dtype=np.float32))
    _write(output, np.ones((3, 3, 3), dtype=np.float32))
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def test_all_zero_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    _write(reference, np.ones((4, 4, 4), dtype=np.float32))
    _write(output, np.zeros((4, 4, 4), dtype=np.float32))
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def _unexpected_label_output(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    _write(reference, np.ones((4, 4, 4), dtype=np.float32))
    data = np.ones((4, 4, 4), dtype=np.int16)
    data[0, 0, 0] = 7
    _write(output, data)
    return reference, output


def test_unknown_labels_warn_without_override(tmp_path: Path, monkeypatch):
    case = _segmentation_case()
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_TRACK", {})
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_CASE", {})
    warnings: list[str] = []
    monkeypatch.setattr(
        validation.logger, "warning", lambda message: warnings.append(message)
    )
    reference, output = _unexpected_label_output(tmp_path)
    validate_case_output(case, output, reference)
    assert warnings and "unexpected label" in warnings[0]


def test_override_makes_labels_strict(tmp_path: Path, monkeypatch):
    case = _segmentation_case()
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_TRACK", {case.slug: [0, 1, 2]})
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_CASE", {})
    reference, output = _unexpected_label_output(tmp_path)
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def test_override_accepts_matching_labels(tmp_path: Path, monkeypatch):
    case = _segmentation_case()
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_TRACK", {case.slug: [0, 1, 7]})
    monkeypatch.setattr(validation, "_ALLOWED_LABELS_BY_CASE", {})
    reference, output = _unexpected_label_output(tmp_path)
    validate_case_output(case, output, reference)


def test_orientation_only_affine_difference_warns(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    _write(reference, reference_data)
    flipped = np.diag([-1.0, -1.0, 1.0, 1.0])
    _write(output, reference_data.astype(np.int16), affine=flipped)
    warnings: list[str] = []
    validate_case_output(case, output, reference, warn=warnings.append)
    assert warnings and "orientation" in warnings[0]


def test_orientation_only_with_translation_warns(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    reference_affine = np.eye(4)
    reference_affine[:3, 3] = [10.0, 20.0, 30.0]
    _write(reference, reference_data, affine=reference_affine)
    # LPS -> RAS relabels the world frame: the flipped axes' translation flips too.
    flipped = np.diag([-1.0, -1.0, 1.0, 1.0])
    _write(output, reference_data.astype(np.int16), affine=flipped @ reference_affine)
    warnings: list[str] = []
    validate_case_output(case, output, reference, warn=warnings.append)
    assert warnings and "orientation" in warnings[0]


def test_inconsistent_sign_translation_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    reference_affine = np.eye(4)
    reference_affine[:3, 3] = [10.0, 20.0, 30.0]
    _write(reference, reference_data, affine=reference_affine)
    # Linear part is flipped but the translation sign does not follow the axes,
    # so this is a genuine geometry mismatch, not an orientation relabel.
    inconsistent = np.eye(4)
    inconsistent[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    inconsistent[:3, 3] = [10.0, 20.0, 30.0]
    _write(output, reference_data.astype(np.int16), affine=inconsistent)
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def test_translation_mismatch_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    _write(reference, reference_data)
    shifted = np.eye(4)
    shifted[:3, 3] = [1.0, 0.0, 0.0]
    _write(output, reference_data.astype(np.int16), affine=shifted)
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)


def test_scale_mismatch_rejected(tmp_path: Path):
    case = _segmentation_case()
    reference = tmp_path / "ref.nii.gz"
    output = tmp_path / "out.nii.gz"
    reference_data = np.ones((4, 4, 4), dtype=np.float32)
    _write(reference, reference_data)
    scaled = np.diag([2.0, 1.0, 1.0, 1.0])
    _write(output, reference_data.astype(np.int16), affine=scaled)
    with pytest.raises(ValidationError):
        validate_case_output(case, output, reference)
