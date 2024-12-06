from __future__ import annotations

import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from loguru import logger

from brats.core.docker import run_container
from brats.utils.algorithm_config import load_algorithms
from brats.constants import OUTPUT_NAME_SCHEMA, Algorithms, Task
from brats.utils.data_handling import InferenceSetup

# Remove the default logger and add one with level INFO
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
)


class BraTSAlgorithm(ABC):
    """
    This class serves as the basis for all BraTS algorithms. It provides a common interface and implements the logic for single and batch inference.
    """

    def __init__(
        self,
        algorithm: Algorithms,
        algorithms_file_path: Path,
        task: Task,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        # inference device setup
        self.force_cpu = force_cpu
        self.cuda_devices = cuda_devices

        self.task = task
        self.algorithm_list = load_algorithms(file_path=algorithms_file_path)
        # save algorithm identifier for logging etc.
        self.algorithm_key = algorithm.value
        # data for selected algorithm
        self.algorithm = self.algorithm_list[algorithm.value]

        logger.info(
            f"Instantiated {self.__class__.__name__} with algorithm: {self.algorithm_key} by {self.algorithm.meta.authors}"
        )

    @abstractmethod
    def _standardize_single_inputs(
        self, data_folder: Path, subject_id: str, inputs: dict[str, Path | str]
    ) -> None:
        """
        Standardize the input data to match the requirements of the selected algorithm.
        """
        pass

    @abstractmethod
    def _standardize_batch_inputs(
        self, data_folder: Path, subjects: list[Path], input_name_schema: str
    ) -> None:
        """
        Standardize the input data to match the requirements of the selected algorithm.
        """
        pass

    def _process_single_output(
        self, tmp_output_folder: Path | str, subject_id: str, output_file: Path
    ) -> None:
        """
        Process the output of a single inference run and save it in the specified file.

        Args:
            tmp_output_folder (Path | str): Folder with the algorithm output
            subject_id (str): Subject ID of the output
            output_file (Path): Path to the desired output file
        """
        # rename output
        if self.task == Task.MISSING_MRI:
            # Missing MRI has no fixed names since the missing modality differs and is included in the name
            algorithm_output = Path(tmp_output_folder).iterdir().__next__()
        else:
            algorithm_output = Path(tmp_output_folder) / OUTPUT_NAME_SCHEMA[
                self.task
            ].format(subject_id=subject_id)

        # ensure path exists and rename output to the desired path
        output_file = Path(output_file).absolute()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(algorithm_output, output_file)

    def _process_batch_output(
        self,
        tmp_output_folder: Path | str,
        output_folder: Path,
        mapping: dict[str, str],
    ) -> None:
        """
        Process the outputs of a batch inference run and save them in the specified folder.

        Args:
            tmp_output_folder (Path | str): Folder with the algorithm outputs
            output_folder (Path): Folder to save the outputs
            mapping (dict[str, str]): Mapping from internal to external subject names
        """
        # move outputs and change name back to initially provided one
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        for internal_name, external_name in mapping.items():
            if self.task == Task.MISSING_MRI:
                # Missing MRI has no fixed names since the missing modality differs and is included in the name
                algorithm_output = (
                    Path(tmp_output_folder).glob(f"*{internal_name}*").__next__()
                )
                try:
                    modality = algorithm_output.name.split("-")[-1].split(".")[0]
                except IndexError:
                    logger.warning(
                        f"Could not extract modality from {algorithm_output.name}"
                    )
                    modality = None
                output_file = (
                    output_folder
                    / f"{external_name}{'-' + modality if modality else ''}.nii.gz"
                )
            else:
                algorithm_output = Path(tmp_output_folder) / OUTPUT_NAME_SCHEMA[
                    self.task
                ].format(subject_id=internal_name)
                output_file = output_folder / f"{external_name}.nii.gz"
            shutil.move(algorithm_output, output_file)

    def _infer_single(
        self,
        inputs: dict[str, Path | str],
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """
        Perform a single inference run with the provided inputs and save the output in the specified file.

        Args:
            inputs (dict[str, Path  |  str]): Input Images for the task
            output_file (Path | str): File to save the output
            log_file (Optional[Path  |  str], optional): Log file with extra information. Defaults to None.
        """
        with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):
            logger.info(f"Performing single inference")

            # the id here is arbitrary
            subject_id = self.algorithm.run_args.input_name_schema.format(id=0)

            self._standardize_single_inputs(
                data_folder=tmp_data_folder,
                subject_id=subject_id,
                inputs=inputs,
            )

            run_container(
                algorithm=self.algorithm,
                data_path=tmp_data_folder,
                output_path=tmp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )
            self._process_single_output(
                tmp_output_folder=tmp_output_folder,
                subject_id=subject_id,
                output_file=output_file,
            )
            logger.info(f"Saved output to: {Path(output_file).absolute()}")

    def _infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Optional[Path | str] = None,
    ):
        """Perform a batch inference run with the provided inputs and save the outputs in the specified folder.

        Args:
            data_folder (Path | str): Folder with the input data
            output_folder (Path | str): Folder to save the outputs
            log_file (Optional[Path  |  str], optional): Log file with extra information. Defaults to None.
        """
        with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):

            # find subjects
            subjects = [f for f in Path(data_folder).iterdir() if f.is_dir()]
            logger.info(
                f"Found {len(subjects)} subjects: {', '.join([s.name for s in subjects][:5])} {' ...' if len(subjects) > 5 else '' }"
            )
            # map to brats names
            internal_external_name_map = self._standardize_batch_inputs(
                data_folder=tmp_data_folder,
                subjects=subjects,
                input_name_schema=self.algorithm.run_args.input_name_schema,
            )
            logger.info(f"Standardized input names to match algorithm requirements.")

            # run inference in container
            run_container(
                algorithm=self.algorithm,
                data_path=tmp_data_folder,
                output_path=tmp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )

            self._process_batch_output(
                tmp_output_folder=tmp_output_folder,
                output_folder=output_folder,
                mapping=internal_external_name_map,
            )

            logger.info(f"Saved outputs to: {Path(output_folder).absolute()}")
