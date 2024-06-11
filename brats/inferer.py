from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from brats.constants import BRATS_INPUT_NAME_SCHEMA, AlgorithmKeys, Device
from brats.data import load_algorithms
from brats.docker import _run_docker

logger = logging.getLogger(__name__)


class Inferer:
    def __init__(
        self,
        algorithm: AlgorithmKeys = AlgorithmKeys.BraTS23_faking_it,
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

        # load algorithm data
        self.algorithm_list = load_algorithms()
        self.algorithm_key = algorithm.value
        self.algorithm = self.algorithm_list[algorithm.value]
        logger.info(
            f"Instantiated Inferer class with algorithm: {algorithm.value} by {self.algorithm.meta.authors}"
        )

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
            Meaning, e.g.:
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

    def infer_single(
        self,
        t1c: Path | str,
        t1n: Path | str,
        t2f: Path | str,
        t2w: Path | str,
        output_file: Path | str,
    ):
        # setup temp input folder with the provided images
        temp_data_folder = Path(tempfile.mkdtemp())
        temp_output_folder = Path(tempfile.mkdtemp())
        try:
            # for a single inference we use a fixed subject id since it is renamed to the desired output afterwards
            subject_id = BRATS_INPUT_NAME_SCHEMA.format(id=0)
            self._standardize_subject_inputs(
                data_folder=temp_data_folder,
                subject_id=subject_id,
                t1c=t1c,
                t1n=t1n,
                t2f=t2f,
                t2w=t2w,
            )

            # self._run_docker(data_folder=temp_data_folder, output_folder=temp_output_folder)
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

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
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
