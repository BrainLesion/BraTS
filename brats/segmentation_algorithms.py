from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from brats.brats_algorithm import BraTSAlgorithm
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
from brats.utils import standardize_segmentation_inputs


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
        """
        Standardize the input data to match the requirements of the selected algorithm.

        Args:
            data_folder (Path): Path to the data folder
            subject_id (str): Subject ID
            inputs (dict[str, Path | str]): Dictionary with the input data
        """
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

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
    ):
        return self._infer_batch(data_folder, output_folder, log_file)


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
