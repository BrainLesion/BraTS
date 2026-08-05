from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import nibabel as nib
from loguru import logger
from rich.console import Console


def remove_tmp_folder(folder: Path) -> None:
    """Remove a temporary folder and log a warning if it fails.

    Args:
        folder (Path): Path to the folder to be removed
    """
    try:
        shutil.rmtree(folder)
    except PermissionError as e:
        logger.warning(
            f"Failed to remove temporary folder {folder}. This is most "
            f"likely caused by bad permission management of the docker "
            f"container. \nError: {e}"
        )
    except FileNotFoundError as e:
        logger.warning(f"Failed to delete folder {folder}. {e}")


def add_log_file_handler(log_file: Path | str) -> int:
    """
    Add a log file handler to the logger.

    Args:
        log_file (Path | str): Path to the log file

    Returns:
        int: The logger id
    """
    log_file = Path(log_file)
    logger_id = logger.add(log_file, level="DEBUG", catch=True)
    logger.info(
        f"Logging console logs and further debug information to: {log_file.absolute()}"
    )

    return logger_id


@contextmanager
def InferenceSetup(
    log_file: Optional[Path | str] = None,
) -> Generator[tuple[Path, Path], None, None]:
    """
    Context manager for setting up the inference process. Creates temporary
    data and output folders and adds a log file handler if requested.

    When no log_file is provided, a temporary log file is created automatically.
    If an error occurs during inference, the temporary log is preserved and its
    location is printed to stderr so users can inspect it for debugging.
    On success, the temporary log is cleaned up.

    Yields:
        (data folder, output folder) (Tuple[Path, Path]): Two temporary
        folders (data folder, output folder)
    """
    if log_file is not None:
        log_path = Path(log_file)
        logger_id = add_log_file_handler(log_path)
    else:
        fd, tmp_log_path = tempfile.mkstemp(suffix=".log", prefix="brats_")
        os.close(fd)
        log_path = Path(tmp_log_path)
        logger_id = add_log_file_handler(log_path)

    tmp_data_folder = Path(tempfile.mkdtemp(prefix="data_"))
    tmp_output_folder = Path(tempfile.mkdtemp(prefix="output_"))

    error_occurred = False
    try:
        yield tmp_data_folder, tmp_output_folder
    except Exception:
        error_occurred = True
        raise
    finally:
        remove_tmp_folder(tmp_data_folder)
        remove_tmp_folder(tmp_output_folder)

        logger.remove(logger_id)

        if error_occurred and log_file is None:
            Console(stderr=True).print(
                f"[red]An error occurred during inference.[/red] "
                f"[yellow]Log file saved to:[/yellow] [bold]{log_path.absolute()}[/bold]"
            )
        elif not error_occurred and log_file is None:
            log_path.unlink(missing_ok=True)


def input_sanity_check(
    t1n: Optional[Path | str] = None,
    t1c: Optional[Path | str] = None,
    t2f: Optional[Path | str] = None,
    t2w: Optional[Path | str] = None,
    mask: Optional[Path | str] = None,
) -> None:
    """
    Check if input images have the default shape (240, 240, 155) and log a warning if not.
    Supports different input combinations for segmentation and inpainting tasks.

    Args:
        t1n (Path | str): T1n image path (required for segmentation and inpainting)
        t1c (Path | str, optional): T1c image path (required for segmentation)
        t2f (Path | str, optional): T2f image path (required for segmentation)
        t2w (Path | str, optional): T2w image path (required for segmentation)
        mask (Path | str, optional): Mask image path (required for inpainting)
    """

    # Filter out None values to only include provided images
    images = {
        "t1n": t1n,
        "t1c": t1c,
        "t2f": t2f,
        "t2w": t2w,
        "mask": mask,
    }

    # Load and check shapes
    shapes = {
        label: nib.load(img).shape for label, img in images.items() if img is not None  # type: ignore[attr-defined]
    }

    assert shapes, "No input images provided. At least one image is required."

    if any(shape != (240, 240, 155) for shape in shapes.values()):
        logger.warning(
            "Input images do not have the default shape (240, 240, 155). "
            "This might cause issues with some algorithms and could lead to "
            "errors."
        )
        logger.warning(f"Image shapes: {shapes}")
        logger.warning(
            "If your data is not preprocessed yet, consider using our preprocessing package: https://github.com/BrainLesion/preprocessing"
        )
