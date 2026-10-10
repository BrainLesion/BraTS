"""End-to-end verification of every registered containerized algorithm.

Excluded from the default test run; opt in with ``-m integration``. Use
``-m smoke`` for the curated, sample-runner-friendly subset.
"""

from __future__ import annotations

import pytest

from scripts.verify.cases import SMOKE_CASE_IDS, build_cases
from scripts.verify.models import Status
from scripts.verify.runner import run_case

pytestmark = pytest.mark.integration

CASES = build_cases()


def _param(case):
    marks = [pytest.mark.smoke] if case.case_id in SMOKE_CASE_IDS else []
    return pytest.param(case, id=case.case_id, marks=marks)


@pytest.mark.parametrize("case", [_param(case) for case in CASES])
def test_verify_case(case, verify_options, verify_records, sample_mode) -> None:
    if sample_mode and case.case_id not in SMOKE_CASE_IDS:
        pytest.skip("not part of the downloadable sample subset")

    result = run_case(case, verify_options)
    verify_records.append(result)

    if result.status in (Status.OK, Status.EXPECTED_FAIL):
        return
    if result.status.is_skipped:
        pytest.skip(result.message or result.status.value)
    pytest.fail(f"{result.status.value}: {result.message}", pytrace=False)
