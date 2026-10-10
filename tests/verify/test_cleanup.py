"""Unit tests for exact-artifact cleanup."""

from __future__ import annotations

from pathlib import Path

from scripts.verify.cleanup import (
    dir_size_bytes,
    existing_weight_folders,
    new_weight_folders,
    remove_path,
)


def _touch(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_existing_and_new_weight_folders(tmp_path: Path):
    record = "13382922"
    _touch(tmp_path / f"{record}_v1.0.0" / "weights.pt")
    before = existing_weight_folders(record, tmp_path)
    assert len(before) == 1

    _touch(tmp_path / f"{record}_v1.1.0" / "weights.pt")
    new = new_weight_folders(record, before, tmp_path)
    assert new == [tmp_path / f"{record}_v1.1.0"]


def test_new_weight_folders_ignores_preexisting(tmp_path: Path):
    record = "42"
    _touch(tmp_path / f"{record}_v1.0.0" / "weights.pt")
    before = existing_weight_folders(record, tmp_path)
    assert new_weight_folders(record, before, tmp_path) == []


def test_remove_path_reports_reclaimed_bytes(tmp_path: Path):
    target = tmp_path / "folder"
    _touch(target / "a.bin", b"x" * 100)
    _touch(target / "b.bin", b"y" * 50)
    assert dir_size_bytes(target) == 150
    assert remove_path(target) == 150
    assert not target.exists()
