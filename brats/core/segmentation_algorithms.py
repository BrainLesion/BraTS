from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from brats.core.brats_algorithm import BraTSAlgorithm
from brats.constants import (
    ADULT_GLIOMA_SEGMENTATION_ALGORITHMS,
    AFRICA_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    METASTASES_SEGMENTATION_ALGORITHMS,
    PEDIATRIC_SEGMENTATION_ALGORITHMS,
    AdultGliomaAlgorithms,
    AfricaAlgorithms,
    Algorithms,
    MeningiomaAlgorithms,
    MetastasesAlgorithms,
    PediatricAlgorithms,
    Task,
)
from brats.utils.data_handling import input_sanity_check


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

    def _standardize_single_inputs(
        self, data_folder: Path, subject_id: str, inputs: Dict[str, Path | str]
    ) -> None:
        """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@subject_id.
        Example:
            Meaning, e.g. for adult glioma:
                BraTS-GLI-00000-000 \n
                ┣ BraTS-GLI-00000-000-t1c.nii.gz \n
                ┣ BraTS-GLI-00000-000-t1n.nii.gz \n
                ┣ BraTS-GLI-00000-000-t2f.nii.gz \n
                ┗ BraTS-GLI-00000-000-t2w.nii.gz \n

        Args:
            data_folder (Path): Parent folder where the subject folder will be created
            subject_id (str): Subject ID to be used for the folder and filenames
            inputs (Dict[str, Path | str]): Dictionary with the input images
        """

        subject_folder = data_folder / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)
        # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
        t1c, t1n, t2f, t2w = inputs["t1c"], inputs["t1n"], inputs["t2f"], inputs["t2w"]
        try:
            shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
            shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
            shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
            shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            logger.error(
                "If you use batch processing please ensure the input files are in the correct format, i.e.:\n A/A-t1c.nii.gz, A/A-t1n.nii.gz, A/A-t2f.nii.gz, A/A-t2w.nii.gz"
            )
            sys.exit(1)

        # sanity check inputs
        input_sanity_check(t1c=t1c, t1n=t1n, t2f=t2f, t2w=t2w)

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
            # TODO Add support for .nii files

            self._standardize_single_inputs(
                data_folder=data_folder,
                subject_id=internal_name,
                inputs={
                    "t1c": subject / f"{subject.name}-t1c.nii.gz",
                    "t1n": subject / f"{subject.name}-t1n.nii.gz",
                    "t2f": subject / f"{subject.name}-t2f.nii.gz",
                    "t2w": subject / f"{subject.name}-t2w.nii.gz",
                },
            )
        return internal_external_name_map

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

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
    ) -> None:
        """Perform segmentation on a batch of subjects with the provided images and save the results to the output folder. \n
        Requires the following structure:\n
        data_folder\n
        ┣ A\n
        ┃ ┣ A-t1c.nii.gz\n
        ┃ ┣ A-t1n.nii.gz\n
        ┃ ┣ A-t2f.nii.gz\n
        ┃ ┗ A-t2w.nii.gz\n
        ┣ B\n
        ┃ ┣ B-t1c.nii.gz\n
        ┃ ┣ ...\n


        Args:
            data_folder (Path | str): Folder containing the subjects with required structure
            output_folder (Path | str): Output folder to save the segmentations
            log_file (Path | str, optional): Save logs to this file
        """
        return self._infer_batch(
            data_folder=data_folder, output_folder=output_folder, log_file=log_file
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
