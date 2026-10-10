"""Zero-disk registry reachability preflight (``-m network``).

Verifies that every referenced container image resolves on the registry without
pulling it. Registered known failures are allowed to be missing.
"""

from __future__ import annotations

import pytest

from scripts.verify.cases import build_cases
from scripts.verify.registry import manifest_inspect

pytestmark = pytest.mark.network

CASES = build_cases()
_IMAGES = sorted({case.docker_image for case in CASES})
_KNOWN_FAILURE_IMAGES = {
    case.docker_image for case in CASES if case.known_failure_reason
}


@pytest.mark.parametrize("image", _IMAGES)
def test_image_reachable(image: str) -> None:
    result = manifest_inspect(image)
    if result.state == "offline":
        pytest.skip(f"registry unreachable: {result.message}")
    if result.state == "pullable":
        return
    if result.state == "private":
        pytest.skip("private image (auth required)")
    if image in _KNOWN_FAILURE_IMAGES:
        return
    pytest.fail(f"image not reachable: {result.message}", pytrace=False)
