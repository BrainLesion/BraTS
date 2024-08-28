from __future__ import annotations

from contextlib import contextmanager
import shutil
import sys
from pathlib import Path
import tempfile
from typing import Dict, Generator, List, Optional, Tuple

import nibabel as nib
from loguru import logger


def standardize_segmentation_inputs(
    data_folder: Path,
    subject_id: str,
    t1c: Path | str,
    t1n: Path | str,
    t2f: Path | str,
    t2w: Path | str,
):
    """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@internal_name.
        Meaning, e.g. for adult glioma:
            BraTS-GLI-00000-000 \n
            ┣ BraTS-GLI-00000-000-t1c.nii.gz \n
            ┣ BraTS-GLI-00000-000-t1n.nii.gz \n
            ┣ BraTS-GLI-00000-000-t2f.nii.gz \n
            ┗ BraTS-GLI-00000-000-t2w.nii.gz \n

    Args:
        data_folder (Path): Parent folder where the subject folder will be created
        subject_id (str): Subject ID to be used for the folder and filenames
        t1c (Path | str): T1c image path
        t1n (Path | str): T1n image path
        t2f (Path | str): T2f image path
        t2w (Path | str): T2w image path
    """

    subject_folder = data_folder / subject_id
    subject_folder.mkdir(parents=True, exist_ok=True)
    # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
    try:
        shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
        shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
        shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
        shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")
    except FileNotFoundError as e:
        logger.error(f"Error while standardizing files: {e}")
        # logger.error(
        #     "If you use batch processing please ensure the input files are in the correct format, i.e.:\n A/A-t1c.nii.gz, A/A-t1n.nii.gz, A/A-t2f.nii.gz, A/A-t2w.nii.gz"
        # )
        sys.exit(1)

    # sanity check inputs
    input_sanity_check(t1c=t1c, t1n=t1n, t2f=t2f, t2w=t2w)


def standardize_inpainting_inputs(
    data_folder: Path,
    subject_id: str,
    t1n: Path | str,
    mask: Path | str,
):

    subject_folder = data_folder / subject_id
    subject_folder.mkdir(parents=True, exist_ok=True)
    # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
    try:
        shutil.copy(t1n, subject_folder / f"{subject_id}-t1n-voided.nii.gz")
        shutil.copy(mask, subject_folder / f"{subject_id}-mask.nii.gz")
    except FileNotFoundError as e:
        logger.error(f"Error while standardizing files: {e}")
        sys.exit(1)

    # sanity check inputs
    input_sanity_check(t1n=t1n, mask=mask)


def standardize_segmentation_inputs_list(
    subjects: List[Path], tmp_data_folder: Path, input_name_schema: str
) -> Dict[str, str]:
    """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @tmp_data_folder/@subject_id.

    Args:
        subjects (List[Path]): List of subject folders, each with a t1c, t1n, t2f, t2w image in standard format
        tmp_data_folder (Path): Parent folder where the subject folders will be created
        input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

    Returns:
        Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
    """
    internal_external_name_map = {}
    for i, subject in enumerate(subjects):
        internal_name = input_name_schema.format(id=i)
        internal_external_name_map[internal_name] = subject.name
        # TODO Add support for .nii files

        standardize_segmentation_inputs(
            data_folder=tmp_data_folder,
            subject_id=internal_name,
            t1c=subject / f"{subject.name}-t1c.nii.gz",
            t1n=subject / f"{subject.name}-t1n.nii.gz",
            t2f=subject / f"{subject.name}-t2f.nii.gz",
            t2w=subject / f"{subject.name}-t2w.nii.gz",
        )
    return internal_external_name_map


def remove_tmp_folder(folder: Path):
    """Remove a temporary folder and log a warning if it fails.

    Args:
        folder (Path): Path to the folder to be removed
    """
    try:
        shutil.rmtree(folder)
    except PermissionError as e:
        logger.warning(
            f"Failed to remove temporary folder {folder}. This is most likely caused by bad permission management of the docker container. \nError: {e}"
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
) -> Generator[Tuple[Path, Path], None, None]:
    """
    Context manager that provides two temporary folders for input and output data and ensures cleanup afterward.

    Yields:
        (data folder, output folder) (Tuple[Path, Path]): Two temporary folders (data folder, output folder)
    """
    if log_file is not None:
        logger_id = add_log_file_handler(log_file)

    tmp_data_folder = Path(tempfile.mkdtemp(prefix="data_"))
    tmp_output_folder = Path(tempfile.mkdtemp(prefix="output_"))

    try:
        yield tmp_data_folder, tmp_output_folder
    finally:
        remove_tmp_folder(tmp_data_folder)
        remove_tmp_folder(tmp_output_folder)

        if log_file is not None:
            logger.remove(logger_id)


def input_sanity_check(
    t1n: Path | str,
    t1c: Optional[Path | str] = None,
    t2f: Optional[Path | str] = None,
    t2w: Optional[Path | str] = None,
    mask: Optional[Path | str] = None,
):
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
        label: nib.load(img).shape for label, img in images.items() if img is not None
    }

    if any(shape != (240, 240, 155) for shape in shapes.values()):
        logger.warning(
            "Input images do not have the default shape (240, 240, 155). This might cause issues with some algorithms and could lead to errors."
        )
        logger.warning(f"Image shapes: {shapes}")
        logger.warning(
            "If your data is not preprocessed yet, consider using our preprocessing package: https://github.com/BrainLesion/preprocessing"
        )
