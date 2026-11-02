"""Exact-artifact teardown for the verification suite.

Cleanup is identity-based: only the image we pulled, only the weight folders
that appeared during this run, and only the temp dir we created. No before/after
directory snapshots are used, so a concurrent install is never removed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from brats.constants import ADDITIONAL_FILES_FOLDER

DockerRunner = Callable[[list[str]], object]


def weights_pattern(record_id: str) -> str:
    """Return the glob pattern for a Zenodo record's extracted folder."""
    return f"{record_id}_v*.*.*"


def existing_weight_folders(
    record_id: str, folder: Path = ADDITIONAL_FILES_FOLDER
) -> set[Path]:
    """Return the weight folders currently present for a record."""
    if not folder.exists():
        return set()
    return set(folder.glob(weights_pattern(record_id)))


def new_weight_folders(
    record_id: str,
    before: set[Path],
    folder: Path = ADDITIONAL_FILES_FOLDER,
) -> list[Path]:
    """Return the weight folders that appeared since ``before`` was captured."""
    return sorted(existing_weight_folders(record_id, folder) - before)


def dir_size_bytes(path: Path) -> int:
    """Return the total size in bytes of a file or folder tree."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def remove_path(path: Path) -> int:
    """Delete a file or folder, returning the number of bytes reclaimed.

    Failures are logged (e.g. root-owned files written by ``requires_root``
    containers) rather than silently ignored, so leaks stay visible.
    """
    size = dir_size_bytes(path)
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning(f"Failed to remove {path}: {e}")
    return size


def _default_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=180, check=False
    )


def remove_image(image_id: str, *, runner: Optional[DockerRunner] = None) -> bool:
    """Remove a local image by its exact id.

    Returns:
        bool: True if the image was removed or already absent
    """
    docker_runner = runner or _default_runner
    proc = docker_runner(["docker", "rmi", image_id])
    if getattr(proc, "returncode", 1) == 0:
        return True
    stderr = (getattr(proc, "stderr", "") or "").lower()
    return "no such image" in stderr
