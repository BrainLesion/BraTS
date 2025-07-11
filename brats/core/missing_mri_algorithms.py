from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Union

from loguru import logger

from brats.core.brats_algorithm import BraTSAlgorithm
from brats.constants import MISSING_MRI_ALGORITHMS, MissingMRIAlgorithms, Task
from brats.utils.data_handling import input_sanity_check


class MissingMRI(BraTSAlgorithm):

    def __init__(
        self,
        algorithm: MissingMRIAlgorithms = MissingMRIAlgorithms.BraTS24_1,
        cuda_devices: str | None = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=MISSING_MRI_ALGORITHMS,
            task=Task.MISSING_MRI,
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

        try:
            for modality, path in inputs.items():
                shutil.copy(
                    path,
                    subject_folder
                    / f"{subject_id}{subject_modality_separator}{modality}.nii.gz",
                )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            raise

        # sanity check inputs
        input_sanity_check(
            t1c=inputs.get("t1c"),
            t1n=inputs.get("t1n"),
            t2f=inputs.get("t2f"),
            t2w=inputs.get("t2w"),
        )

    def _standardize_batch_inputs(
        self,
        data_folder: Path,
        subjects: List[Path],
        input_name_schema: str,
    ) -> Dict[str, str]:
        """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @tmp_data_folder/@subject_id.

        Args:
            subjects (List[Path]): List of subject folders, each with a t1c, t1n, t2f, t2w image in standard format
            data_folder (Path): Parent folder where the subject folders will be created
            input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

        Returns:
            Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
        """
        internal_external_name_map = {}
        for i, subject in enumerate(subjects):
            internal_name = input_name_schema.format(id=i)
            internal_external_name_map[internal_name] = subject.name

            # find relevant files in the subject folder
            possible_inputs = {
                "t1c": subject / f"{subject.name}-t1c.nii.gz",
                "t1n": subject / f"{subject.name}-t1n.nii.gz",
                "t2f": subject / f"{subject.name}-t2f.nii.gz",
                "t2w": subject / f"{subject.name}-t2w.nii.gz",
            }
            valid_inputs = {k: v for k, v in possible_inputs.items() if v.exists()}
            assert (
                len(valid_inputs) == 3
            ), "Exactly 3 inputs are required to perform synthesis of the missing modality"

            self._standardize_single_inputs(
                data_folder=data_folder,
                subject_id=internal_name,
                inputs=valid_inputs,
                subject_modality_separator=self.algorithm.run_args.subject_modality_separator,
            )
        return internal_external_name_map

    def infer_single(
        self,
        output_file: Path | str,
        t1c: Optional[Union[Path, str]] = None,
        t1n: Optional[Union[Path, str]] = None,
        t2f: Optional[Union[Path, str]] = None,
        t2w: Optional[Union[Path, str]] = None,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """
        Perform synthesis of the missing modality for a single subject with the provided images and save the result to the output file.

        Note:
            Exactly 3 input modalities are required to perform synthesis of the missing modality.

        Args:
            output_file (Path | str): Output file to save the synthesized image
            t1c (Optional[Union[Path, str]], optional): Path to the T1c image. Defaults to None.
            t1n (Optional[Union[Path, str]], optional): Path to the T1n image. Defaults to None.
            t2f (Optional[Union[Path, str]], optional): Path to the T2f image. Defaults to None.
            t2w (Optional[Union[Path, str]], optional): Path to the T2w image. Defaults to None.
            log_file (Optional[Path | str], optional): Save logs to this file. Defaults to None
        """

        inputs = {"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w}
        # filter out None values
        inputs = {k: v for k, v in inputs.items() if v is not None}

        # assert exactly 3 inputs are given (to compute the missing one)
        assert (
            len(inputs) == 3
        ), "Exactly 3 inputs are required to perform synthesis of the missing modality"

        self._infer_single(
            inputs=inputs,
            output_file=output_file,
            log_file=log_file,
        )

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
    ) -> None:
        """Perform synthesis on a batch of subjects with the provided images and save the results to the output folder. \n

        Requires the following structure (if e.g. t2f should be synthesized):\n
        data_folder\n
        ┣ A\n
        ┃ ┣ A-t1c.nii.gz\n
        ┃ ┣ A-t1n.nii.gz\n
        ┃ ┗ A-t2w.nii.gz\n
        ┣ B\n
        ┃ ┣ B-t1c.nii.gz\n
        ┃ ┣ ...\n


        Args:
            data_folder (Path | str): Folder containing the subjects with required structure
            output_folder (Path | str): Output folder to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        return self._infer_batch(
            data_folder=data_folder, output_folder=output_folder, log_file=log_file
        )
