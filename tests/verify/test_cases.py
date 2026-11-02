"""Unit tests for case discovery and registry-to-dataset mapping."""

from __future__ import annotations

from collections import Counter

from brats.constants import Task
from brats.registry import iter_algorithms
from scripts.verify.cases import SMOKE_CASE_IDS, build_cases, slug_for
from scripts.verify.datasets import track_slugs


def test_build_cases_counts():
    cases = build_cases()
    assert len(cases) == 79
    assert len({case.track for case in cases}) == 10


def test_case_ids_are_unique():
    cases = build_cases()
    ids = [case.case_id for case in cases]
    assert len(ids) == len(set(ids))


def test_pre_post_track_has_both_variants():
    cases = build_cases()
    pre_post = [c for c in cases if c.slug == "adult_glioma_pre_post"]
    assert len(pre_post) == 14
    timepoints = Counter(c.treatment_timepoint for c in pre_post)
    assert timepoints == {"pre": 7, "post": 7}
    for case in pre_post:
        assert case.kwargs["treatment_timepoint"] == case.treatment_timepoint


def test_known_failures_are_marked():
    cases = build_cases()
    expected = [c for c in cases if c.known_failure_reason]
    assert len(expected) == 4
    assert all("goat" in c.docker_image for c in expected)


def test_every_case_has_registered_dataset_and_task():
    cases = build_cases()
    valid_tasks = {"segmentation", "inpainting", "missing_mri"}
    for case in cases:
        assert case.task in valid_tasks
        assert case.dataset_key == case.slug
        assert case.docker_image


def test_smoke_subset_is_valid():
    ids = {case.case_id for case in build_cases()}
    assert ids >= SMOKE_CASE_IDS


def test_task_vocabulary_matches_source():
    source_tasks = {task.value.lower() for task in Task}
    case_tasks = {case.task for case in build_cases()}
    assert case_tasks <= source_tasks


def test_every_registered_track_has_dataset_spec():
    slugs = {slug_for(record) for record in iter_algorithms()}
    assert slugs == set(track_slugs())
