from __future__ import annotations

import logging
import os
import shutil
import tempfile
from abc import ABC
from pathlib import Path
from typing import Optional

from brats.constants import (
    ADULT_GLIOMA_SEGMENTATION_ALGORITHMS,
    ADULT_GLIOMA_INPUT_NAME_SCHEMA,
    MENINGIOMA_INPUT_NAME_SCHEMA,
    MENINGIOMA_SEGMENTATION_ALGORITHMS,
    AdultGliomaAlgorithmKeys,
    Device,
    MeningiomaAlgorithmKeys,
)
from brats.data import load_algorithms
from brats.docker import _run_docker

logger = logging.getLogger(__name__)


class BraTSInferer(ABC):
    def __init__(
        self,
        device: Device = Device.AUTO,
        cuda_devices: Optional[str] = "0",
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z",
        )

        self.device = device
        self.cuda_devices = cuda_devices

    def _standardize_subject_inputs(
        self,
        data_folder: Path,
        subject_id: str,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
    ):
        """Standardize the input images for a single subject to match requirements of all algorithms.
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

        shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
        shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
        shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
        shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")

    def _infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
        subject_format: str,
    ):
        # setup temp input folder with the provided images
        temp_data_folder = Path(tempfile.mkdtemp())
        temp_output_folder = Path(tempfile.mkdtemp())
        try:
            # for a single inference we use a fixed subject id since it is renamed to the desired output afterwards
            subject_id = subject_format.format(id=0)
            self._standardize_subject_inputs(
                data_folder=temp_data_folder,
                subject_id=subject_id,
                t1c=t1c,
                t1n=t1n,
                t2f=t2f,
                t2w=t2w,
            )

            _run_docker(
                algorithm=self.algorithm,
                data_path=temp_data_folder,
                output_path=temp_output_folder,
                cuda_devices=self.cuda_devices,
            )
            print(os.listdir(temp_output_folder))
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

        # infer
        _run_docker(
            algorithm=self.algorithm,
            data_path=data_folder,
            output_path=output_folder,
            cuda_devices=self.cuda_devices,
        )

        # rename outputs


class AdultGliomaInferer(BraTSInferer):

    def __init__(
        self,
        algorithm: AdultGliomaAlgorithmKeys = AdultGliomaAlgorithmKeys.BraTS23_glioma_faking_it,
        device: Device = Device.AUTO,
        cuda_devices: Optional[str] = "0",
    ):
        super().__init__(device=device, cuda_devices=cuda_devices)
        # load algorithm data
        self.algorithm_list = load_algorithms(
            file_path=ADULT_GLIOMA_SEGMENTATION_ALGORITHMS
        )
        self.algorithm_key = algorithm.value
        self.algorithm = self.algorithm_list[algorithm.value]
        logger.info(
            f"Instantiated AdultGliomaInferer class with algorithm: {algorithm.value} by {self.algorithm.meta.authors}"
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
            subject_format=ADULT_GLIOMA_INPUT_NAME_SCHEMA,
        )

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        self._infer_batch(data_folder=data_folder, output_folder=output_folder)


class MeningiomaInferer(BraTSInferer):

    def __init__(
        self,
        algorithm: MeningiomaAlgorithmKeys = MeningiomaAlgorithmKeys.BraTS23_meningioma_nvauto,
        device: Device = Device.AUTO,
        cuda_devices: Optional[str] = "0",
    ):
        super().__init__(device=device, cuda_devices=cuda_devices)

        # TODO: make this more dry since it is the same as in AdultGliomaInferer etc.
        # load algorithm data
        self.algorithm_list = load_algorithms(
            file_path=MENINGIOMA_SEGMENTATION_ALGORITHMS
        )
        self.algorithm_key = algorithm.value
        self.algorithm = self.algorithm_list[algorithm.value]
        logger.info(
            f"Instantiated MeningiomaInferer class with algorithm: {algorithm.value} by {self.algorithm.meta.authors}"
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
