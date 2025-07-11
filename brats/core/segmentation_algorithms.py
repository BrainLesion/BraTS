from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

from loguru import logger

from brats.constants import (
    ADULT_GLIOMA_POST_TREATMENT_SEGMENTATION_ALGORITHMS,
    ADULT_GLIOMA_PRE_TREATMENT_SEGMENTATION_ALGORITHMS,
    AFRICA_SEGMENTATION_ALGORITHMS,
    GOAT_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    METASTASES_SEGMENTATION_ALGORITHMS,
    PEDIATRIC_SEGMENTATION_ALGORITHMS,
    AdultGliomaPostTreatmentAlgorithms,
    AdultGliomaPreTreatmentAlgorithms,
    AfricaAlgorithms,
    Algorithms,
    GoATAlgorithms,
    MeningiomaAlgorithms,
    MetastasesAlgorithms,
    PediatricAlgorithms,
    Task,
)
from brats.core.brats_algorithm import BraTSAlgorithm
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
        self,
        data_folder: Path,
        subject_id: str,
        inputs: Dict[str, Path | str],
        subject_modality_separator: str,
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
            subject_modality_separator (str): Separator between the subject ID and the modality
        """

        subject_folder = data_folder / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)
        # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
        try:
            for modality, path in inputs.items():
                shutil.copy(
                    path,
                    subject_folder
                    / f"{subject_id}{subject_modality_separator}{modality}.nii.gz",
                )
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            logger.error(
                "If you use batch processing please ensure the input files are in the correct format, i.e.:\n A/A-t1c.nii.gz, A/A-t1n.nii.gz, A/A-t2f.nii.gz, A/A-t2w.nii.gz"
            )
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
        only_t1c: bool = False,
    ) -> Dict[str, str]:
        """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @tmp_data_folder/@subject_id.

        Args:
            subjects (List[Path]): List of subject folders, each with a t1c, t1n, t2f, t2w image in standard format
            data_folder (Path): Parent folder where the subject folders will be created
            input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge
            only_t1c (bool, optional): If True, only the t1c image will be used. Defaults to False.

        Returns:
            Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
        """
        internal_external_name_map = {}
        for i, subject in enumerate(subjects):
            internal_name = input_name_schema.format(id=i)
            internal_external_name_map[internal_name] = subject.name

            inputs = {
                "t1c": subject / f"{subject.name}-t1c.nii.gz",
            }
            if not only_t1c:
                inputs["t1n"] = subject / f"{subject.name}-t1n.nii.gz"
                inputs["t2f"] = subject / f"{subject.name}-t2f.nii.gz"
                inputs["t2w"] = subject / f"{subject.name}-t2w.nii.gz"

            self._standardize_single_inputs(
                data_folder=data_folder,
                subject_id=internal_name,
                inputs=inputs,
                subject_modality_separator=self.algorithm.run_args.subject_modality_separator,
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


class AdultGliomaPreTreatmentSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on adult glioma pre treatment MRI data.

    Args:
        algorithm (AdultGliomaPreTreatmentAlgorithms, optional): Select an algorithm. Defaults to AdultGliomaPreTreatmentAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AdultGliomaPreTreatmentAlgorithms = AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=ADULT_GLIOMA_PRE_TREATMENT_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class AdultGliomaPostTreatmentSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on adult glioma post treatment MRI data.

    Args:
        algorithm (AdultGliomaPostTreatmentAlgorithms, optional): Select an algorithm. Defaults to AdultGliomaPostTreatmentAlgorithms.BraTS24_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AdultGliomaPostTreatmentAlgorithms = AdultGliomaPostTreatmentAlgorithms.BraTS24_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=ADULT_GLIOMA_POST_TREATMENT_SEGMENTATION_ALGORITHMS,
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

    def _standardize_batch_inputs(
        self,
        data_folder: Path,
        subjects: List[Path],
        input_name_schema: str,
        only_t1c: bool = False,
    ) -> Dict[str, str]:
        """Standardize the input images for a list of subjects to match requirements of all algorithms and save them in @tmp_data_folder/@subject_id.

        Args:
            subjects (List[Path]): List of subject folders, each with a t1c, t1n, t2f, t2w image in standard format
            data_folder (Path): Parent folder where the subject folders will be created
            input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

        Returns:
            Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
        """
        only_t1c = self.algorithm.meta.year == 2024
        return super()._standardize_batch_inputs(
            data_folder=data_folder,
            subjects=subjects,
            input_name_schema=input_name_schema,
            only_t1c=only_t1c,
        )

    def infer_single(
        self,
        output_file: Path | str,
        t1c: Union[Path, str] = None,
        t1n: Optional[Union[Path, str]] = None,
        t2f: Optional[Union[Path, str]] = None,
        t2w: Optional[Union[Path, str]] = None,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """
        Perform segmentation on a single subject with the provided images and save the result to the output file.

        Note:
            Unlike other segmentation challenges, not all modalities are required for all meningioma segmentation algorithms.
            Differences by year:

            - **2024**: Only `t1c` is required and used by the algorithms.
            - **2023**: All (`t1c`, `t1n`, `t2f`, `t2w`) are required.


        Args:
            output_file (Path | str): Output file to save the segmentation.
            t1c (Union[Path, str]): Path to the T1c image. Defaults to None.
            t1n (Optional[Union[Path, str]], optional): Path to the T1n image. Defaults to None.
            t2f (Optional[Union[Path, str]], optional): Path to the T2f image. Defaults to None.
            t2w (Optional[Union[Path, str]], optional): Path to the T2w image. Defaults to None.
            log_file (Optional[Path | str], optional): Save logs to this file. Defaults to None
        """
        inputs = {"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w}
        # filter out None values
        inputs = {k: v for k, v in inputs.items() if v is not None}

        year = self.algorithm.meta.year
        if year == 2024:
            if "t1c" not in inputs or len(inputs) > 1:
                raise ValueError(
                    (
                        "Only the T1C modality is required and used by the 2024 meningioma segmentation challenge and its algorithms. "
                        "Please provide only the T1C modality - Aborting to avoid confusion."
                    )
                )
        elif year == 2023:
            if len(inputs) != 4:
                raise ValueError(
                    (
                        "All modalities (t1c, t1n, t2f, t2w) are required for the 2023 meningioma segmentation challenge and its algorithms. "
                        "Please provide all modalities"
                    )
                )
        else:
            raise NotImplementedError(
                f"Invalid algorithm {year=} .Only 2023 and 2024 are supported as of now"
            )
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
        """
        Perform segmentation on a batch of subjects with the provided images and save the results to the output folder. \n

        Note:
            Unlike other segmentation challenges, not all modalities are required for all meningioma segmentation algorithms.
            Differences by year:

            - **2024**: Only `t1c` is required and used by the algorithms.
            - **2023**: All (`t1c`, `t1n`, `t2f`, `t2w`) are required.

        Requires the following structure (example for 2023, only t1c for 2024!):\n
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
            data_folder=data_folder,
            output_folder=output_folder,
            log_file=log_file,
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
    """Provides algorithms from the BraTSAfrica challenge

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
    """Provides algorithms from the Brain Metastases Segmentation challenge

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


class GoATSegmenter(SegmentationAlgorithm):
    """Provides algorithms from the BraTS Generalizability Across Tumors (BraTS-GoAT)

    Args:
        algorithm (GoATAlgorithms, optional): Select an algorithm. Defaults to GoATAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: GoATAlgorithms = GoATAlgorithms.BraTS24_1,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=GOAT_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )
