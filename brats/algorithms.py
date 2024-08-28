from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, Optional


from loguru import logger

from brats.algorithm_config import load_algorithms
from brats.constants import (
    ADULT_GLIOMA_SEGMENTATION_ALGORITHMS,
    AFRICA_SEGMENTATION_ALGORITHMS,
    METASTASES_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    PEDIATRIC_SEGMENTATION_ALGORITHMS,
    INPAINTING_ALGORITHMS,
    OUTPUT_NAME_SCHEMA,
    AdultGliomaAlgorithms,
    Algorithms,
    AfricaAlgorithms,
    InpaintingAlgorithms,
    MetastasesAlgorithms,
    MeningiomaAlgorithms,
    PediatricAlgorithms,
    Task,
)
from brats.docker import run_docker
from brats.utils import (
    InferenceSetup,
    standardize_inpainting_inputs,
    standardize_segmentation_inputs,
)
from abc import ABC, abstractmethod


# Remove the default logger and add one with level INFO
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
)


class BraTSAlgorithm(ABC):
    """This class serves as the basis for all BraTS algorithms. It provides a common interface and implements the logic for single and batch inference."""

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
    def _standardize_inputs(
        self, data_folder: Path, subject_id: str, inputs: dict[str, Path | str]
    ) -> None:
        """Standardize the input data to match the requirements of the selected algorithm."""
        pass

    def _log_algorithm_info(self):
        """Log information about the selected algorithm."""
        logger.opt(colors=True).info(
            f"Running algorithm: <light-green>{self.algorithm.meta.challenge} [{self.algorithm.meta.rank} place]</>"
        )
        logger.opt(colors=True).info(
            f"<blue>(Paper)</> Consider citing the corresponding paper: {self.algorithm.meta.paper} by {self.algorithm.meta.authors}"
        )

    def _process_output(
        self, tmp_output_folder: Path | str, subject_id: str, output_file: Path
    ) -> None:
        # rename output
        algorithm_output = Path(tmp_output_folder) / OUTPUT_NAME_SCHEMA[
            self.task
        ].format(subject_id=subject_id)

        # ensure path exists and rename output to the desired path
        output_file = Path(output_file).absolute()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(algorithm_output, output_file)

    def _infer_single(
        self,
        inputs: dict[str, Path | str],
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ):
        with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):
            logger.info(f"Performing single inference")

            # the id here is arbitrary
            subject_id = self.algorithm.run_args.input_name_schema.format(id=0)

            self._standardize_inputs(
                data_folder=tmp_data_folder,
                subject_id=subject_id,
                inputs=inputs,
            )

            self._log_algorithm_info()
            run_docker(
                algorithm=self.algorithm,
                data_path=tmp_data_folder,
                output_path=tmp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )
            self._process_output(
                tmp_output_folder=tmp_output_folder,
                subject_id=subject_id,
                output_file=output_file,
            )
            logger.info(f"Saved output to: {Path(output_file).absolute()}")

    # def _infer_batch(
    #     self,
    #     data_folder: Path | str,
    #     output_folder: Path | str,
    #     log_file: Optional[Path | str] = None,
    # ):

    #     with InferenceSetup(log_file=log_file) as (tmp_data_folder, tmp_output_folder):
    #         data_folder = Path(data_folder)

    #         output_folder.mkdir(parents=True, exist_ok=True)
    #         self._log_algorithm_info()
    #         # find subjects
    #         subjects = [f for f in data_folder.iterdir() if f.is_dir()]
    #         logger.info(
    #             f"Found {len(subjects)} subjects: {', '.join([s.name for s in subjects][:5])} {' ...' if len(subjects) > 5 else '' }"
    #         )
    #         # map to brats names
    #         internal_external_name_map = standardize_segmentation_inputs_list(
    #             subjects=subjects,
    #             tmp_data_folder=tmp_data_folder,
    #             input_name_schema=self.algorithm.run_args.input_name_schema,
    #         )
    #         logger.info(f"Standardized input names to match algorithm requirements.")

    #         # run inference in container
    #         run_docker(
    #             algorithm=self.algorithm,
    #             data_path=tmp_data_folder,
    #             output_path=tmp_output_folder,
    #             cuda_devices=self.cuda_devices,
    #             force_cpu=self.force_cpu,
    #         )

    # # move outputs and change name back to initially provided one
    # output_folder = Path(output_folder)
    # for internal_name, external_name in internal_external_name_map.items():
    #     segmentation = Path(tmp_output_folder) / f"{internal_name}.nii.gz"
    #     output_file = output_folder / f"{external_name}.nii.gz"
    #     shutil.move(segmentation, output_file)

    # logger.info(f"Saved outputs to: {output_folder.absolute()}")


class SegmentationAlgorithm(BraTSAlgorithm):
    """This class provides algorithms to perform tumor segmentation on MRI data. It is the base class for all segmentation algorithms and provides the common interface for single and batch inference."""

    def __init__(
        self,
        algorithm: Algorithms,
        algorithms_file_path: Path,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=algorithms_file_path,
            task=Task.SEGMENTATION,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )

    def _standardize_inputs(
        self, data_folder: Path, subject_id: str, inputs: Dict[str, Path | str]
    ) -> None:
        """Standardize the input data to match the requirements of the selected algorithm."""
        standardize_segmentation_inputs(
            data_folder=data_folder,
            subject_id=subject_id,
            t1c=inputs["t1c"],
            t1n=inputs["t1n"],
            t2f=inputs["t2f"],
            t2w=inputs["t2w"],
        )

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """Perform segmentation on a single subject with the provided images and save the result to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            t1n (Path | str): Path to the T1n image
            t2f (Path | str): Path to the T2f image
            t2w (Path | str): Path to the T2w image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        self._infer_single(
            inputs={"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w},
            output_file=output_file,
            log_file=log_file,
        )


class AdultGliomaSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on adult glioma MRI data.

    Args:
        algorithm (AdultGliomaAlgorithms, optional): Select an algorithm. Defaults to AdultGliomaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AdultGliomaAlgorithms = AdultGliomaAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=ADULT_GLIOMA_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class MeningiomaSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on adult meningioma MRI data.

    Args:
        algorithm (MeningiomaAlgorithms, optional): Select an algorithm. Defaults to MeningiomaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: MeningiomaAlgorithms = MeningiomaAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=MENINGIOMA_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class PediatricSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on pediatric MRI data

    Args:
        algorithm (PediatricAlgorithms, optional): Select an algorithm. Defaults to PediatricAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: PediatricAlgorithms = PediatricAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=PEDIATRIC_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class AfricaSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on data from the BraTSAfrica challenge

    Args:
        algorithm (AfricaAlgorithms, optional): Select an algorithm. Defaults to AfricaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AfricaAlgorithms = AfricaAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=AFRICA_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class MetastasesSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on data from the Brain Metastases Segmentation challenge

    Args:
        algorithm (MetastasesAlgorithms, optional): Select an algorithm. Defaults to MetastasesAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: MetastasesAlgorithms = MetastasesAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=METASTASES_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


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

    def _standardize_inputs(
        self, data_folder: Path, subject_id: str, inputs: dict[str, Path | str]
    ) -> None:
        """Standardize the input data to match the requirements of the selected algorithm."""
        standardize_inpainting_inputs(
            data_folder=data_folder,
            subject_id=subject_id,
            t1n=inputs["t1n"],
            mask=inputs["mask"],
        )

    def infer_single(
        self,
        t1n: Path | str,
        mask: Path | str,
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """Perform segmentation on a single subject with the provided images and save the result to the output file.

        Args:
            t1n (Path | str): Path to the T1n image
            mask (Path | str): Path to the mask image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        self._infer_single(
            inputs={"t1n": t1n, "mask": mask},
            output_file=output_file,
            log_file=log_file,
        )
