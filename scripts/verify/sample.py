"""Optional download of a small sample dataset for the verification suite.

The upstream ``BrainLesion/tutorials`` repository ships one de-identified
glioblastoma subject for the segmentation and inpainting tracks. This module
mirrors those files into the directory layout the dataset registry expects, so
the ``smoke`` subset can run on a host without a full ``~/brats_data`` tree.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

# Pinned commit of BrainLesion/tutorials so downloaded sample data is stable.
DATA_REF = "99c5841fe8985e4004d3d5394ca9590a914e24e7"
DATA_BASE_URL = "https://raw.githubusercontent.com/BrainLesion/tutorials"
SUBJECT_ID = "BraTS-GLI-00001-000"

_SEGMENTATION_MODALITIES = ("t1c", "t1n", "t2f", "t2w")
_INPAINTING_FILES = ("mask", "t1n-voided")


class SampleDataError(RuntimeError):
    """Raised when the sample data cannot be downloaded."""


def _download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` atomically (skip if present)."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "brats-verify"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: SIM117
            with open(tmp, "wb") as file:
                shutil.copyfileobj(response, file)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise SampleDataError(f"Failed to download {url}: {e}") from e

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise SampleDataError(f"Downloaded empty file from {url}")
    tmp.replace(destination)


def _source_url(dataset: str, suffix: str) -> str:
    return (
        f"{DATA_BASE_URL}/{DATA_REF}/BraTS/data/{dataset}/"
        f"{SUBJECT_ID}/{SUBJECT_ID}-{suffix}.nii.gz"
    )


def download_sample_data(data_root: Path) -> Path:
    """Populate ``data_root`` with the downloadable sample subject.

    The files are placed under the per-track directories the dataset registry
    resolves (``adult_glioma``, ``missing_mri`` and ``inpainting``).

    Args:
        data_root (Path): Root folder to populate

    Returns:
        Path: ``data_root`` for convenient chaining
    """
    data_root = Path(data_root)
    segmentation_dir = data_root / "adult_glioma" / SUBJECT_ID

    for modality in _SEGMENTATION_MODALITIES:
        _download(
            _source_url("segmentation", modality),
            segmentation_dir / f"{SUBJECT_ID}-{modality}.nii.gz",
        )

    missing_mri_dir = data_root / "missing_mri" / SUBJECT_ID
    for modality in ("t1c", "t1n", "t2f"):
        _download(
            _source_url("segmentation", modality),
            missing_mri_dir / f"{SUBJECT_ID}-{modality}.nii.gz",
        )

    inpainting_dir = data_root / "inpainting" / SUBJECT_ID
    for suffix in _INPAINTING_FILES:
        _download(
            _source_url("inpainting", suffix),
            inpainting_dir / f"{SUBJECT_ID}-{suffix}.nii.gz",
        )

    return data_root
