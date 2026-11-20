"""Unit tests for the terminal/JSON reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.verify.cases import build_cases
from scripts.verify.models import CaseResult, Status
from scripts.verify.reporting import render_summary, summarize, write_json


def _result(status: Status, case_id: str = "inpainting.BraTS23_1") -> CaseResult:
    case = next(c for c in build_cases() if c.case_id == case_id)
    return CaseResult(case=case, status=status, duration_s=1.0, message="")


def test_summarize_counts():
    results = [
        _result(Status.OK),
        _result(Status.EXPECTED_FAIL),
        _result(Status.RUN_FAIL),
        _result(Status.GPU_BUSY_SKIPPED),
    ]
    assert summarize(results) == {
        "total": 4,
        "ok": 1,
        "expected_fail": 1,
        "failed": 1,
        "skipped": 1,
    }


def test_write_json_roundtrip(tmp_path: Path):
    results = [_result(Status.OK), _result(Status.GPU_BUSY_SKIPPED)]
    path = write_json(results, tmp_path / "nested" / "results.json")
    payload = json.loads(path.read_text())
    assert payload["summary"]["total"] == 2
    assert len(payload["results"]) == 2
    assert payload["results"][0]["case_id"] == "inpainting.BraTS23_1"


def test_render_summary_mentions_total():
    lines: list[str] = []
    render_summary(lines.append, [_result(Status.OK)])
    assert any("TOTAL" in line for line in lines)
    assert any("BraTS verification summary" in line for line in lines)
