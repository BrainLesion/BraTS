from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import docker
from brats.constants import AlgorithmKeys, Device
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

        vols = [
            v for v in [data_folder, weights_folder, output_folder] if v is not None
        ]
        volumes = {
            Path(v).absolute(): {
                "bind": f"/mlcube_io{i}",
                "mode": "rw",
            }
            for i, v in enumerate(vols)
        }

        logger.info(f"{' Starting inference ':-^80}")
        logger.info(
            f"Algorithm: {self.algorithm_key} | Docker image: {self.algorithm.image}"
        )
        logger.info(f"Consider citing the corresponding paper: {self.algorithm.paper}")
        logger.info(
            f">> Note: Outputs below are streamed from the container and subject to the respective author's logging"
        )

        args = (
            f"--data_path=/mlcube_io0 --weights=/mlcube_io1 --output_path=/mlcube_io2"
            if weights_folder is not None
            else f"--data_path=/mlcube_io0 --output_path=/mlcube_io1"
        )

        extra_args = {}
        if not self.algorithm.requires_root:
            # run the container as the current user to ensure written files are always owned by the user
            extra_args["user"] = f"{os.getuid()}:{os.getgid()}"

        # Run the container
        container = client.containers.run(
            image=self.algorithm.image,
            volumes=volumes,
            # TODO: how to support CPU?
            device_requests=[
                docker.types.DeviceRequest(
                    device_ids=[self.cuda_devices], capabilities=[["gpu"]]
                )
            ],
            # Constant params for the docker execution dictated by the mlcube format
            command=f"infer {args}",
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

    def infer_single(
        self,
        t1: Path | str,
        t1c: Path | str,
        t2: Path | str,
        flair: Path | str,
        output: Path | str,
    ):
        pass

    def infer_batch(self, data_folder: Path | str, output_folder: Path | str):
        self._infer(data_folder=data_folder, output_folder=output_folder)
