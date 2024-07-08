from __future__ import annotations

import shutil
import signal
import sys
from pathlib import Path
from typing import Dict, List

from loguru import logger


def standardize_subject_inputs(
    data_folder: Path,
    subject_id: str,
    t1c: Path | str,
    t1n: Path | str,
    t2f: Path | str,
    t2w: Path | str,
):
    """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@subject_id.
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

    # os.symlink would be more efficient but can cause issues on windows
    # TODO: use symlink on unix systems
    try:
        shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
        shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
        shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
        shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")
    except FileNotFoundError as e:
        logger.error(f"Error while standardizing files: {e}")
        logger.error(
            "Please make sure the input files are in the correct format, i.e.:\n A/A-t1c.nii.gz, A/A-t1n.nii.gz, A/A-t2f.nii.gz, A/A-t2w.nii.gz"
        )
        sys.exit(1)


def standardize_subjects_inputs_list(
    subjects: List[Path], temp_data_folder: Path, input_name_schema: str
) -> Dict[str, str]:
    """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @temp_data_folder/@subject_id.

    Args:
        subjects (List[Path]): List of subject folders, each with a t1c, t1n, t2f, t2w image in standard format
        temp_data_folder (Path): Parent folder where the subject folders will be created
        input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

    Returns:
        Dict[str, str]: Dictionary mapping internal subject_id (in standardized format) to subject name provided by user
    """
    subject_id_name_map = {}
    for i, subject in enumerate(subjects):
        subject_id = input_name_schema.format(id=i)
        subject_id_name_map[subject_id] = subject.name
        # TODO Add support for .nii files
        standardize_subject_inputs(
            data_folder=temp_data_folder,
            subject_id=subject_id,
            t1c=subject / f"{subject.name}-t1c.nii.gz",
            t1n=subject / f"{subject.name}-t1n.nii.gz",
            t2f=subject / f"{subject.name}-t2f.nii.gz",
            t2w=subject / f"{subject.name}-t2w.nii.gz",
        )
    return subject_id_name_map


# def handle_signals():
#     """Handle signals to exit gracefully and log the signal received."""

#     def signal_handler(sig, frame):
#         signame = signal.Signals(sig).name
#         logger.error(f"Received signal {sig} ({signame}), exiting...")
#         sys.exit(0)

#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
