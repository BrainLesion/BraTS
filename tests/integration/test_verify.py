"""End-to-end verification of every registered containerized algorithm.

Excluded from the default test run; opt in with ``-m integration``. Use
``-m smoke`` for the curated, fast subset.
"""

from __future__ import annotations

from collections import Counter

import pytest

from scripts.verify.cases import SMOKE_CASE_IDS, build_cases
from scripts.verify.models import Status
from scripts.verify.runner import run_case

pytestmark = pytest.mark.integration

CASES = build_cases()

# Cases that share a multi-GB image (e.g. pre/post pairs) would otherwise remove
# and immediately re-pull it. Keep the image while a later executed case still
# needs it and release it on the last one; the session-level cleanup remains the
# safety net (e.g. when a later case is deselected with -k/-m).
_IMAGE_USES = Counter(case.docker_image for case in CASES)


def _param(case):
    marks = [pytest.mark.smoke] if case.case_id in SMOKE_CASE_IDS else []
    return pytest.param(case, id=case.case_id, marks=marks)


@pytest.mark.parametrize("case", [_param(case) for case in CASES])
def test_verify_case(case, verify_options, verify_records) -> None:
    keep_image = _IMAGE_USES[case.docker_image] > 1
    _IMAGE_USES[case.docker_image] -= 1
    result = run_case(case, verify_options, keep_image=keep_image)
    verify_records.append(result)

    if result.status in (Status.OK, Status.EXPECTED_FAIL):
        return
    if result.status.is_skipped:
        pytest.skip(result.message or result.status.value)
    pytest.fail(f"{result.status.value}: {result.message}", pytrace=False)
