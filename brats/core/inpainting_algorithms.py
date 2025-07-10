from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Dict, Optional

from loguru import logger

from brats.core.brats_algorithm import BraTSAlgorithm
from brats.constants import INPAINTING_ALGORITHMS, InpaintingAlgorithms, Task
from brats.utils.data_handling import input_sanity_check


class Inpainter(BraTSAlgorithm):

    def __init__(
        self,
        algorithm: InpaintingAlgorithms = InpaintingAlgorithms.BraTS23_1,
        cuda_devices: str | None = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=INPAINTING_ALGORITHMS,
            task=Task.INPAINTING,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )

    def _standardize_single_inputs(
        self,
        data_folder: Path,
        subject_id: str,
        inputs: dict[str, Path | str],
        subject_modality_separator: str,
    ) -> None:
        """
        Standardize the input data to match the requirements of the selected algorithm.

        Args:
            data_folder (Path): Path to the data folder
            subject_id (str): Subject ID
            inputs (dict[str, Path | str]): Dictionary with the input data
            subject_modality_separator (str): Separator between the subject ID and the modality
        """

        subject_folder = data_folder / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)
        # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
        t1n, mask = inputs["t1n"], inputs["mask"]
        try:
            shutil.copy(
                t1n,
                subject_folder
                / f"{subject_id}{subject_modality_separator}t1n-voided.nii.gz",
            )
            shutil.copy(
                mask,
                subject_folder / f"{subject_id}{subject_modality_separator}mask.nii.gz",
            )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            raise

        # sanity check inputs
        input_sanity_check(t1n=t1n, mask=mask)

    def _standardize_batch_inputs(
        self, data_folder: Path, subjects: list[Path], input_name_schema: str
    ) -> Dict[str, str]:
        """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @tmp_data_folder/@subject_id.

        Args:
            subjects (List[Path]): List of subject folders, each with a voided t1n and a mask image
            data_folder (Path): Parent folder where the subject folders will be created
            input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

        Returns:
            Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
        """
        internal_external_name_map = {}
        for i, subject in enumerate(subjects):
            internal_name = input_name_schema.format(id=i)
            internal_external_name_map[internal_name] = subject.name
            # TODO Add support for .nii files

            self._standardize_single_inputs(
                data_folder=data_folder,
                subject_id=internal_name,
                inputs={
                    "t1n": subject / f"{subject.name}-t1n-voided.nii.gz",
                    "mask": subject / f"{subject.name}-mask.nii.gz",
                },
                subject_modality_separator=self.algorithm.run_args.subject_modality_separator,
            )
        return internal_external_name_map

    def infer_single(
        self,
        t1n: Path | str,
        mask: Path | str,
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """Perform inpainting task on a single subject with the provided images and save the result to the output file.

        Args:
            t1n (Path | str): Path to the voided T1n image
            mask (Path | str): Path to the mask image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        self._infer_single(
            inputs={"t1n": t1n, "mask": mask},
            output_file=output_file,
            log_file=log_file,
        )

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
    ) -> None:
        """Perform inpainting on a batch of subjects with the provided images and save the results to the output folder. \n
        Requires the following structure:\n
        data_folder\n
        ┣ A\n
        ┃ ┣ A-t1n-voided.nii.gz\n
        ┃ ┣ A-mask.nii.gz\n
        ┣ B\n
        ┃ ┣ B-t1n-voided.nii.gz\n
        ┃ ┣ B-mask.nii.gz\n
        ┣ C ...\n


        Args:
            data_folder (Path | str): Folder containing the subjects with required structure
            output_folder (Path | str): Output folder to save the segmentations
            log_file (Path | str, optional): Save logs to this file
        """
        return self._infer_batch(
            data_folder=data_folder, output_folder=output_folder, log_file=log_file
        )
