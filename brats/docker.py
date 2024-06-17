from __future__ import annotations

import logging
import os
from pathlib import Path

import docker
from brats.data import AlgorithmData
from brats.utils import check_model_weights
from halo import Halo
from rich.progress import Progress


logger = logging.getLogger(__name__)
client = docker.from_env()


def show_progress(tasks, line, progress: Progress):
    if line["status"] == "Downloading":
        task_key = f'[Download {line["id"]}]'
    elif line["status"] == "Extracting":
        task_key = f'[Extract  {line["id"]}]'
    else:
        return

    if task_key not in tasks.keys():
        tasks[task_key] = progress.add_task(
            f"{task_key}", total=line["progressDetail"]["total"]
        )
    else:
        progress.update(tasks[task_key], completed=line["progressDetail"]["current"])


def _ensure_image(image: str):
    if not client.images.list(name=image):
        logger.info(f"Pulling docker image {image}")
        tasks = {}
        with Progress() as progress:
            resp = client.api.pull(image, stream=True, decode=True)
            for line in resp:
                show_progress(tasks, line, progress)


def _run_docker(
    algorithm: AlgorithmData,
    data_path: Path | str,
    output_path: Path | str,
    cuda_devices: str,
):

    # ensure image is present, if not pull it
    _ensure_image(algorithm.run_args.docker_image)

    # ensure weights are present
    if algorithm.weights is not None:
        weights_path = check_model_weights(record_id=algorithm.weights.record_id)
    else:
        weights_path = None

    # ensure output folder exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Define the volumes expected by the mlcube standard
    # data input: /mlcube_io0
    # additional files (mostly weights): /mlcube_io1
    # output: /mlcube_io2

    volumes = [v for v in [data_path, weights_path, output_path] if v is not None]
    volume_mappings = {
        Path(v).absolute(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, v in enumerate(volumes)
    }

    logger.info(f"{' Starting inference ':-^80}")
    logger.info(f"Docker image: {algorithm.run_args.docker_image}")
    logger.info(f"Consider citing the corresponding paper: {algorithm.meta.paper}")

    command_args = (
        f"--data_path=/mlcube_io0 --{algorithm.weights.param_name}=/mlcube_io1 --output_path=/mlcube_io2"
        if algorithm.weights is not None
        else f"--data_path=/mlcube_io0 --output_path=/mlcube_io1"
    )

    if algorithm.run_args.parameters_file:
        # The algorithms do not seem to actually use the parameters file  for inference but just need it to exist
        # so we create an empty file
        parameters_file = weights_path / "parameters.yaml"
        parameters_file.touch()
        command_args += f" --parameters_file=/mlcube_io1/parameters.yaml"

    extra_args = {}
    if not algorithm.run_args.requires_root:
        # run the container as the current user to ensure written files are always owned by the user
        # also overall better security-wise
        extra_args["user"] = f"{os.getuid()}:{os.getgid()}"

    # Run the container
    container = client.containers.run(
        image=algorithm.run_args.docker_image,
        volumes=volume_mappings,
        # TODO: how to support CPU?
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=[cuda_devices], capabilities=[["gpu"]]
            )
        ],
        # Constant params for the docker execution dictated by the mlcube format
        command=f"infer {command_args}",
        network_mode="none",
        detach=True,
        remove=True,
        shm_size=algorithm.run_args.shm_size,
        **extra_args,
    )

    # capture the output
    container_output = container.attach(
        stdout=True, stderr=True, stream=True, logs=True
    )

    # Display spinner while the container is running
    spinner = Halo(text="Running inference...", spinner="dots")
    spinner.start()

    # Wait for the container to finish
    exit_code = container.wait()
    # Check if the container exited with an error
    if exit_code["StatusCode"] != 0:
        spinner.stop_and_persist(symbol="X", text="Container finished with an error.")
        for line in container_output:
            logger.error(f">> {line.decode('utf-8')}")
        raise Exception("Container finished with an error. See logs above for details.")
    else:
        spinner.stop_and_persist(symbol="✔", text="Inference done.")

    logger.info(f"{' Finished inference ':-^80}")
