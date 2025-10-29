from __future__ import annotations

import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from loguru import logger

from brats.constants import (
    ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS,
    ADULT_GLIOMA_PRE_TREATMENT_SEGMENTATION_ALGORITHMS,
    AFRICA_SEGMENTATION_ALGORITHMS,
    GOAT_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_RT_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    METASTASES_SEGMENTATION_ALGORITHMS,
    PEDIATRIC_SEGMENTATION_ALGORITHMS,
    AdultGliomaPreAndPostTreatmentAlgorithms,
    AdultGliomaPreTreatmentAlgorithms,
    AfricaAlgorithms,
    Algorithms,
    GoATAlgorithms,
    MeningiomaAlgorithms,
    MeningiomaRTAlgorithms,
    MetastasesAlgorithms,
    PediatricAlgorithms,
    Task,
    Backends,
)
from brats.core.brats_algorithm import BraTSAlgorithm
from brats.utils.data_handling import input_sanity_check


class SegmentationAlgorithm(BraTSAlgorithm):
    """This class provides algorithms to perform tumor segmentation on MRI data. It is the base class for all segmentation algorithms and provides the common interface for single and batch inference."""

    def __init__(
        self,
        algorithm: Algorithms,
        algorithms_file_path: Path,
        cuda_devices: str = "0",
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
        inputs: Mapping[str, Path | str],
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

    @abstractmethod
    def infer_single(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
    ) -> None:
        pass


class SegmentationAlgorithmWith4Modalities(SegmentationAlgorithm):
    """Segmentation algorithm that works with 4 modalities (T1c, T1n, T2f, T2w)."""

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
        backend: Optional[Backends] = Backends.DOCKER,
    ) -> None:
        """Perform segmentation on a single subject with the provided images and save the result to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            t1n (Path | str): Path to the T1n image
            t2f (Path | str): Path to the T2f image
            t2w (Path | str): Path to the T2w image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
            backend (Backends, optional): Backend to use for inference. Defaults to Backends.DOCKER.
        """

        self._infer_single(
            inputs={"t1c": t1c, "t1n": t1n, "t2f": t2f, "t2w": t2w},
            output_file=output_file,
            log_file=log_file,
            backend=backend,
        )

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
        backend: Optional[Backends] = Backends.DOCKER,
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
            backend (Backends, optional): Backend to use for inference. Defaults to Backends.DOCKER.
        """

        return self._infer_batch(
            data_folder=data_folder,
            output_folder=output_folder,
            log_file=log_file,
            backend=backend,
        )


class AdultGliomaPreTreatmentSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms to perform tumor segmentation on adult glioma pre treatment MRI data.

    Args:
        algorithm (AdultGliomaPreTreatmentAlgorithms, optional): Select an algorithm. Defaults to AdultGliomaPreTreatmentAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AdultGliomaPreTreatmentAlgorithms = AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=ADULT_GLIOMA_PRE_TREATMENT_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class AdultGliomaPreAndPostTreatmentSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms to perform tumor segmentation on adult glioma pre and post treatment MRI data.

    Args:
        algorithm (AdultGliomaPreAndPostTreatmentAlgorithms, optional): Select an algorithm. Defaults to AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AdultGliomaPreAndPostTreatmentAlgorithms = AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class MeningiomaSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms to perform tumor segmentation on adult meningioma MRI data.

    Args:
        algorithm (MeningiomaAlgorithms, optional): Select an algorithm. Defaults to MeningiomaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: MeningiomaAlgorithms = MeningiomaAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=MENINGIOMA_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class PediatricSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms to perform tumor segmentation on pediatric MRI data

    Args:
        algorithm (PediatricAlgorithms, optional): Select an algorithm. Defaults to PediatricAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: PediatricAlgorithms = PediatricAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=PEDIATRIC_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class AfricaSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms from the BraTSAfrica challenge

    Args:
        algorithm (AfricaAlgorithms, optional): Select an algorithm. Defaults to AfricaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: AfricaAlgorithms = AfricaAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=AFRICA_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class MetastasesSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms from the Brain Metastases Segmentation challenge

    Args:
        algorithm (MetastasesAlgorithms, optional): Select an algorithm. Defaults to MetastasesAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: MetastasesAlgorithms = MetastasesAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=METASTASES_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


class GoATSegmenter(SegmentationAlgorithmWith4Modalities):
    """Provides algorithms from the BraTS Generalizability Across Tumors (BraTS-GoAT)

    Args:
        algorithm (GoATAlgorithms, optional): Select an algorithm. Defaults to GoATAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: GoATAlgorithms = GoATAlgorithms.BraTS25_1A,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=GOAT_SEGMENTATION_ALGORITHMS,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )


### Radio Therapy specific segmenter (only T1C) ###


class MeningiomaRTSegmenter(SegmentationAlgorithm):
    """Provides algorithms to perform tumor segmentation on adult meningioma Radio Therapy MRI data.

    Args:
        algorithm (MeningiomaAlgorithms, optional): Select an algorithm. Defaults to MeningiomaAlgorithms.BraTS23_1.
        cuda_devices (Optional[str], optional): Which cuda devices to use. Defaults to "0".
        force_cpu (bool, optional): Execution will default to GPU, this flag allows forced CPU execution if the algorithm is compatible. Defaults to False.
    """

    def __init__(
        self,
        algorithm: MeningiomaRTAlgorithms = MeningiomaRTAlgorithms.BraTS25_1,
        cuda_devices: str = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=MENINGIOMA_RT_SEGMENTATION_ALGORITHMS,
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
            subjects (List[Path]): List of subject folders, each with a t1c image in standard format
            data_folder (Path): Parent folder where the subject folders will be created
            input_name_schema (str): Schema to be used for the subject folder and filenames depending on the BraTS Challenge

        Returns:
            Dict[str, str]: Dictionary mapping internal name (in standardized format) to external subject name provided by user
        """
        return super()._standardize_batch_inputs(
            data_folder=data_folder,
            subjects=subjects,
            input_name_schema=input_name_schema,
            only_t1c=True,
        )

    def infer_single(
        self,
        t1c: Union[Path, str],
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
        backend: Optional[Backends] = Backends.DOCKER,
    ) -> None:
        """
        Perform segmentation on a single subject with the provided T1C image and save the result to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            output_file (Path | str): Output file to save the segmentation.
            log_file (Optional[Path | str], optional): Save logs to this file. Defaults to None.
            backend (Backends, optional): Backend to use for inference. Defaults to Backends.DOCKER.
        """

        self._infer_single(
            inputs={"t1c": t1c},
            output_file=output_file,
            log_file=log_file,
            backend=backend,
        )

    def infer_batch(
        self,
        data_folder: Path | str,
        output_folder: Path | str,
        log_file: Path | str | None = None,
        backend: Optional[Backends] = Backends.DOCKER,
    ) -> None:
        """
        Perform segmentation on a batch of subjects with the provided T1C images and save the results to the output folder. \n


        Requires the following structure:\n
        data_folder\n
        ┣ A\n
        ┃ ┗ A-t1c.nii.gz\n
        ┣ B\n
        ┃ ┗ B-t1c.nii.gz\n
        ┃ ...\n


        Args:
            data_folder (Path | str): Folder containing the subjects with required structure
            output_folder (Path | str): Output folder to save the segmentations
            log_file (Path | str, optional): Save logs to this file
            backend (Backends, optional): Backend to use for inference. Defaults to Backends.DOCKER.
        """

        return self._infer_batch(
            data_folder=data_folder,
            output_folder=output_folder,
            log_file=log_file,
            backend=backend,
        )
