"""Compact reporting for the verification suite (terminal + JSON)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from scripts.verify.models import CaseResult, Status

_TERMINAL_STATUS = {
    Status.OK: "ok",
    Status.EXPECTED_FAIL: "expected",
    Status.PREFLIGHT_FAIL: "FAIL",
    Status.PULL_FAIL: "FAIL",
    Status.DISK_SKIPPED: "skip",
    Status.GPU_BUSY_SKIPPED: "skip",
    Status.RUN_FAIL: "FAIL",
    Status.INVALID_OUTPUT: "FAIL",
    Status.TIMEOUT: "FAIL",
    Status.ENV_ERROR: "FAIL",
}


def summarize(results: Iterable[CaseResult]) -> dict[str, int]:
    """Return counts of total, ok, expected_fail, failed and skipped."""
    counts = {
        "total": 0,
        "ok": 0,
        "expected_fail": 0,
        "failed": 0,
        "skipped": 0,
    }
    for result in results:
        counts["total"] += 1
        if result.status is Status.OK:
            counts["ok"] += 1
        elif result.status is Status.EXPECTED_FAIL:
            counts["expected_fail"] += 1
        elif result.status.is_failure:
            counts["failed"] += 1
        elif result.status.is_skipped:
            counts["skipped"] += 1
    return counts


def _track_counts(results: list[CaseResult]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case.track, []).append(result)
    return {track: summarize(items) for track, items in sorted(grouped.items())}


def render_summary(write: Callable[[str], None], results: list[CaseResult]) -> None:
    """Write a compact per-track overview plus the failures.

    Args:
        write (Callable[[str], None]): Line sink (e.g. pytest's ``write_line``)
        results (List[CaseResult]): All case results collected so far
    """
    if not results:
        return
    track_counts = _track_counts(results)
    overall = summarize(results)
    write("")
    write("BraTS verification summary")
    write(f"{'track':<34}{'total':>6}{'ok':>6}{'exp':>6}{'fail':>6}{'skip':>6}")
    for track, counts in track_counts.items():
        write(
            f"{track:<34}{counts['total']:>6}{counts['ok']:>6}"
            f"{counts['expected_fail']:>6}{counts['failed']:>6}"
            f"{counts['skipped']:>6}"
        )
    write(
        f"{'TOTAL':<34}{overall['total']:>6}{overall['ok']:>6}"
        f"{overall['expected_fail']:>6}{overall['failed']:>6}{overall['skipped']:>6}"
    )

    failures = [r for r in results if r.status.is_failure]
    if failures:
        write("")
        write(f"Failures ({len(failures)}):")
        for result in failures:
            write(
                f"  {result.case.case_id:<40} {result.status.value}: {result.message}"
            )


def _result_to_dict(result: CaseResult) -> dict[str, object]:
    return {
        "track": result.case.track,
        "slug": result.case.slug,
        "member": result.case.member,
        "case_id": result.case.case_id,
        "class": result.case.class_name,
        "year": result.case.year,
        "image": result.case.docker_image,
        "status": result.status.value,
        "duration_s": round(result.duration_s, 3),
        "message": result.message,
        "output_file": str(result.output_file) if result.output_file else None,
        "log_file": str(result.log_file) if result.log_file else None,
        "image_pulled_by_us": result.image_pulled_by_us,
        "image_id": result.image_id,
        "requires_weights": result.case.requires_weights,
        "warnings": list(result.warnings),
    }


def write_json(results: list[CaseResult], path: Path) -> Path:
    """Persist the collected results as JSON.

    Returns:
        Path: The path the report was written to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarize(results),
        "tracks": _track_counts(results),
        "status_counts": dict(Counter(r.status.value for r in results)),
        "results": [_result_to_dict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
