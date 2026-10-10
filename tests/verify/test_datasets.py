"""Unit tests for the track -> dataset mapping."""

from __future__ import annotations

from pathlib import Path

from scripts.verify.cases import build_cases
from scripts.verify.datasets import (
    missing_inputs,
    resolve_inputs,
    resolve_reference,
    resolve_subject_dir,
)


def _case(slug: str, member: str | None = None, timepoint: str | None = None):
    for case in build_cases():
        if case.slug != slug:
            continue
        if member is not None and not case.member.startswith(member):
            continue
        if timepoint is not None and case.treatment_timepoint != timepoint:
            continue
        return case
    raise AssertionError(f"case not found: {slug} {member} {timepoint}")


def test_four_modality_paths(tmp_path: Path):
    case = _case("pediatric", "BraTS25_1")
    inputs = resolve_inputs(case, tmp_path)
    subject = tmp_path / "pediatric" / "BraTS-PED-00030-000"
    assert inputs["t1c"] == subject / "BraTS-PED-00030-000-t1c.nii.gz"
    assert inputs["t2w"] == subject / "BraTS-PED-00030-000-t2w.nii.gz"


def test_inpainting_paths(tmp_path: Path):
    case = _case("inpainting", "BraTS23_1")
    inputs = resolve_inputs(case, tmp_path)
    subject = tmp_path / "inpainting" / "BraTS-GLI-00001-000"
    assert inputs["t1n"] == subject / "BraTS-GLI-00001-000-t1n-voided.nii.gz"
    assert inputs["mask"] == subject / "BraTS-GLI-00001-000-mask.nii.gz"
    assert resolve_reference(case, inputs) == inputs["t1n"]


def test_missing_mri_has_three_inputs(tmp_path: Path):
    case = _case("missing_mri", "BraTS24_1")
    inputs = resolve_inputs(case, tmp_path)
    assert set(inputs) == {"t1c", "t1n", "t2w"}


def test_adult_glioma_uses_unprefixed_filenames(tmp_path: Path):
    case = _case("adult_glioma_pre_treatment", "BraTS23_1")
    inputs = resolve_inputs(case, tmp_path)
    subject = tmp_path / "adult_glioma" / "BraTS-GLI-00001-000"
    assert inputs["t1c"] == subject / "t1c.nii.gz"
    assert inputs["t2w"] == subject / "t2w.nii.gz"


def test_pre_post_filename_conventions_differ(tmp_path: Path):
    pre = _case("adult_glioma_pre_post", "BraTS25_1", timepoint="pre")
    post = _case("adult_glioma_pre_post", "BraTS25_1", timepoint="post")
    pre_inputs = resolve_inputs(pre, tmp_path)
    post_inputs = resolve_inputs(post, tmp_path)
    assert pre_inputs["t1c"] == (
        tmp_path / "adult_glioma" / "BraTS-GLI-00001-000" / "t1c.nii.gz"
    )
    assert post_inputs["t1c"] == (
        tmp_path
        / "adult_glioma_postop"
        / "BraTS-GLI-02073-100"
        / "BraTS-GLI-02073-100-t1c.nii.gz"
    )


def test_meningioma_rt_uses_underscore_filename(tmp_path: Path):
    case = _case("meningioma_rt", "BraTS25_1")
    inputs = resolve_inputs(case, tmp_path)
    subject = tmp_path / "adult_meningioma_24" / "BraTS-MEN-RT-0010-1"
    assert inputs["t1c"] == subject / "BraTS-MEN-RT-0010-1_t1c.nii.gz"


def test_metastases_year_selects_subject(tmp_path: Path):
    old = _case("metastases", "BraTS23_1")
    new = _case("metastases", "BraTS25_1")
    assert resolve_subject_dir(old, tmp_path) == (
        tmp_path / "metastases" / "BraTS-MET-00013-000"
    )
    assert resolve_subject_dir(new, tmp_path) == (
        tmp_path / "MET25" / "BraTS-MET-00833-000"
    )


def test_pre_post_timepoint_selects_subject(tmp_path: Path):
    pre = _case("adult_glioma_pre_post", "BraTS25_1", timepoint="pre")
    post = _case("adult_glioma_pre_post", "BraTS25_1", timepoint="post")
    assert resolve_subject_dir(pre, tmp_path) == (
        tmp_path / "adult_glioma" / "BraTS-GLI-00001-000"
    )
    assert resolve_subject_dir(post, tmp_path) == (
        tmp_path / "adult_glioma_postop" / "BraTS-GLI-02073-100"
    )


def test_missing_inputs(tmp_path: Path):
    case = _case("pediatric", "BraTS25_1")
    inputs = resolve_inputs(case, tmp_path)
    assert len(missing_inputs(inputs)) == 4
    first = next(iter(inputs.values()))
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"x")
    assert len(missing_inputs(inputs)) == 3
