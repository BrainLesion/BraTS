from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import docker
from brats.constants import AlgorithmKeys, Device, BRATS_INPUT_NAME_SCHEMA
from brats.data import load_algorithms
from brats.utils import check_model_weights

logger = logging.getLogger(__name__)


class Inferer:
    def __init__(
        self,
        algorithm: AlgorithmKeys = AlgorithmKeys.BraTS23_yaziciz,
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
            f"Instantiated Inferer class with algorithm: {algorithm.value} by {[a.name for a in self.algorithm.authors]}"
        )

    def _infer(self, data_folder: Path | str, output_folder: Path | str):

        # ensure weights are present
        if self.algorithm.zenodo_record_id is not None:
            weights_folder = check_model_weights(
                record_id=self.algorithm.zenodo_record_id
            )
        else:
            weights_folder = None

        # ensure output folder exists
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # Initialize the Docker client
        client = docker.from_env()

        # Define the volumes expected by the mlcube standard
        # data input: /mlcube_io0
        # additional files (mostly weights): /mlcube_io1
        # output: /mlcube_io2

        volumes = [
            v for v in [data_folder, weights_folder, output_folder] if v is not None
        ]
        volume_mappings = {
            Path(v).absolute(): {
                "bind": f"/mlcube_io{i}",
                "mode": "rw",
            }
            for i, v in enumerate(volumes)
        }

        logger.info(f"{' Starting inference ':-^80}")
        logger.info(
            f"Algorithm: {self.algorithm_key} | Docker image: {self.algorithm.image}"
        )
        logger.info(f"Consider citing the corresponding paper: {self.algorithm.paper}")
        logger.info(
            f">> Note: Outputs below are streamed from the container and subject to the respective author's logging"
        )

        command_args = (
            f"--data_path=/mlcube_io0 --weights=/mlcube_io1 --output_path=/mlcube_io2"
            if weights_folder is not None
            else f"--data_path=/mlcube_io0 --output_path=/mlcube_io1"
        )

        if self.algorithm.parameters_file:
            parameters_file = data_folder / "parameters.yaml"
            parameters_file.touch()
            command_args += f" --parameters_file="

        extra_args = {}
        if not self.algorithm.requires_root:
            # run the container as the current user to ensure written files are always owned by the user
            extra_args["user"] = f"{os.getuid()}:{os.getgid()}"

        # Run the container
        container = client.containers.run(
            image=self.algorithm.image,
            volumes=volume_mappings,
            # TODO: how to support CPU?
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=[self.cuda_devices], capabilities=[["gpu"]]
                )
            ],
            # Constant params for the docker execution dictated by the mlcube format
            command=f"infer {command_args}",
            network_mode="none",
            detach=True,
            remove=True,
            shm_size=self.algorithm.shm_size,
            **extra_args,
        )

        # Stream the output to the console
        container_output = container.attach(
            stdout=True, stderr=True, stream=True, logs=True
        )
        for line in container_output:
            logger.info(f">> {line.decode('utf-8')}")

        # Wait for the container to finish
        container.wait()
        logger.info(f"{' Finished inference ':-^80}")

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

            self._infer(data_folder=temp_data_folder, output_folder=temp_output_folder)

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
        self._infer(data_folder=data_folder, output_folder=output_folder)

        # rename outputs
