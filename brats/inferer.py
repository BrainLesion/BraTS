from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from brats.algorithm_config import load_algorithms
from brats.constants import (
    ADULT_GLIOMA_INPUT_NAME_SCHEMA,
    ADULT_GLIOMA_SEGMENTATION_ALGORITHMS,
    MENINGIOMA_INPUT_NAME_SCHEMA,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    PEDIATRIC_INPUT_NAME_SCHEMA,
    PEDIATRIC_SEGMENTATION_ALGORITHMS,
    AdultGliomaAlgorithms,
    Algorithms,
    MeningiomaAlgorithms,
    PediatricAlgorithms,
)
from brats.docker import run_docker
from brats.utils import standardize_subject_inputs

# configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S%z",
)


class BraTSInferer(ABC):
    def __init__(
        self,
        algorithm: Algorithms,
        algorithms_file_path: Path,
        cuda_devices: Optional[str] = "0",
        force_cpu: bool = False,
    ):

        # inference device setup
        self.force_cpu = force_cpu
        self.cuda_devices = cuda_devices

        self.algorithm_list = load_algorithms(file_path=algorithms_file_path)
        # save algorithm identifier for logging etc.
        self.algorithm_key = algorithm.value
        # data for selected algorithm
        self.algorithm = self.algorithm_list[algorithm.value]

        logger.info(
            f"Instantiated {self.__class__.__name__} with algorithm: {self.algorithm_key} by {self.algorithm.meta.authors}"
        )

    @abstractmethod
    def infer_single():
        pass

    @abstractmethod
    def infer_batch():
        pass

    def _infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
        subject_format: str,
    ):
        """Perform inference on a single subject with the provided images and save the segmentation to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            t1n (Path | str): Path to the T1n image
            t2f (Path | str): Path to the T2f image
            t2w (Path | str): Path to the T2w image
            output_file (Path | str): Path to save the segmentation
            subject_format (str): Format string for the subject id
        """
        # setup temp input folder with the provided images
        temp_data_folder = Path(tempfile.mkdtemp())
        temp_output_folder = Path(tempfile.mkdtemp())
        try:
            # for a single inference we use a fixed subject id since it is renamed to the desired output afterwards
            subject_id = subject_format.format(id=0)
            standardize_subject_inputs(
                data_folder=temp_data_folder,
                subject_id=subject_id,
                t1c=t1c,
                t1n=t1n,
                t2f=t2f,
                t2w=t2w,
            )

            logger.info(
                f"Running algorithm: {self.algorithm_key} from challenge: {self.algorithm.meta.challenge}"
            )
            run_docker(
                algorithm=self.algorithm,
                data_path=temp_data_folder,
                output_path=temp_output_folder,
                cuda_devices=self.cuda_devices,
                force_cpu=self.force_cpu,
            )
            # rename output
            segmentation = Path(temp_output_folder) / f"{subject_id}.nii.gz"

            # ensure path exists and rename output to the desired path
            output_file = Path(output_file).absolute()
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(segmentation, output_file)

        finally:
            shutil.rmtree(temp_data_folder)
            shutil.rmtree(temp_output_folder)

    def _infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        """Infer all subjects in a folder. requires the following structure:
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
            data_folder (Path | str): _description_
            output_folder (Path | str): _description_
        """

        # map to brats names
        data_folder = Path(data_folder)
        output_folder = Path(output_folder)
        # infer
        run_docker(
            algorithm=self.algorithm,
            data_path=data_folder,
            output_path=output_folder,
            cuda_devices=self.cuda_devices,
            force_cpu=self.force_cpu,
        )

        # rename outputs


class AdultGliomaInferer(BraTSInferer):

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

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
    ):
        """Perform inference on a single subject with the provided images and save the segmentation to the output file.

        Args:
            t1c (Path | str): Path to the T1c image
            t1n (Path | str): Path to the T1n image
            t2f (Path | str): Path to the T2f image
            t2w (Path | str): Path to the T2w image
            output_file (Path | str): Path to save the segmentation
        """
        self._infer_single(
            t1c=t1c,
            t1n=t1n,
            t2f=t2f,
            t2w=t2w,
            output_file=output_file,
            subject_format=ADULT_GLIOMA_INPUT_NAME_SCHEMA,
        )

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        self._infer_batch(data_folder=data_folder, output_folder=output_folder)


class MeningiomaInferer(BraTSInferer):

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

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
    ):
        self._infer_single(
            t1c=t1c,
            t1n=t1n,
            t2f=t2f,
            t2w=t2w,
            output_file=output_file,
            subject_format=MENINGIOMA_INPUT_NAME_SCHEMA,
        )

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        self._infer_batch(data_folder=data_folder, output_folder=output_folder)


class PediatricInferer(BraTSInferer):

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

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
    ):
        self._infer_single(
            t1c=t1c,
            t1n=t1n,
            t2f=t2f,
            t2w=t2w,
            output_file=output_file,
            subject_format=PEDIATRIC_INPUT_NAME_SCHEMA,
        )

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        self._infer_batch(data_folder=data_folder, output_folder=output_folder)
