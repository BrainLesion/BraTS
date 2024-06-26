from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, List

import docker
from brats.data import AlgorithmData
from brats.weights import check_model_weights, get_dummy_weights_path
from halo import Halo
from rich.progress import Progress


logger = logging.getLogger(__name__)
client = docker.from_env()


def _show_docker_pull_progress(tasks: Dict, progress: Progress, line: Dict):
    """Show the progress of a docker pull operation.

    Args:
        tasks (Dict): The tasks to update
        progress (Progress): The progress bar to update
        line (Dict): the next line from docker.client.api.pull stream
    """
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
    """Ensure the docker image is present on the system. If not, pull it.

    Args:
        image (str): The docker image to pull
    """
    if not client.images.list(name=image):
        logger.info(f"Pulling docker image {image}")
        tasks = {}
        with Progress() as progress:
            resp = client.api.pull(image, stream=True, decode=True)
            for line in resp:
                _show_docker_pull_progress(tasks, line, progress)


def _is_cuda_available() -> bool:
    """Check if CUDA is available on the system by trying to run nvidia-smi."""
    try:
        # Attempt to run `nvidia-smi` to check for CUDA.
        # This command should run successfully if NVIDIA drivers are installed and GPUs are present.
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except:
        return False


def _handle_device_requests(
    algorithm: AlgorithmData, cuda_devices: str, force_cpu: bool
) -> List[docker.types.DeviceRequest]:
    """Handle the device requests for the docker container (request cuda or cpu).

    Args:
        algorithm (AlgorithmData): Algorithm data
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
    """
    cuda_available = _is_cuda_available()
    if not cuda_available or force_cpu:
        if not algorithm.run_args.cpu_compatible:
            raise Exception(
                f"No Cuda installation/ GPU was found and the chosen algorithm is not compatible with CPU. Aborting..."
            )
        # empty device requests => run on CPU
        return []
    # request gpu with chosen devices
    return [
        docker.types.DeviceRequest(device_ids=[cuda_devices], capabilities=[["gpu"]])
    ]


def run_docker(
    algorithm: AlgorithmData,
    data_path: Path | str,
    output_path: Path | str,
    cuda_devices: str,
    force_cpu: bool,
):
    """Run a docker container for the provided algorithm.

    Args:
        algorithm (AlgorithmData): The data of the algorithm to run
        data_path (Path | str): The path to the input data
        output_path (Path | str): The path to save the output
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
    """

    # ensure image is present, if not pull it
    _ensure_image(algorithm.run_args.docker_image)

    # ensure weights are present and get path
    if algorithm.weights is not None:
        additional_files_path = check_model_weights(
            record_id=algorithm.weights.record_id
        )
    else:
        # if no weights are directly specified a dummy weights folder will be mounted that is potentially used for parameter files etc.
        additional_files_path = get_dummy_weights_path()

    # ensure output folder exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # data = mlcube_io0, weights = mlcube_io1, output = mlcube_io2
    # TODO: add support for recommended "ro" mount mode for input data
    volume_mappings = {
        Path(v).absolute(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, v in enumerate([data_path, additional_files_path, output_path])
    }

    logger.info(f"{' Starting inference ':-^80}")
    logger.info(f"Docker image: {algorithm.run_args.docker_image}")
    logger.info(f"Consider citing the corresponding paper: {algorithm.meta.paper}")

    # Build command that will be run in the docker container
    command_args = (
        f"--data_path=/mlcube_io0 --{algorithm.weights.param_name}=/mlcube_io1 --output_path=/mlcube_io2"
        if algorithm.weights is not None
        else f"--data_path=/mlcube_io0 --output_path=/mlcube_io2"
    )

    if algorithm.run_args.parameters_file:
        # The algorithms do not seem to actually use the parameters file  for inference but just need it to exist
        # so we create an empty file
        parameters_file = additional_files_path / "parameters.yaml"
        parameters_file.touch()
        command_args += f" --parameters_file=/mlcube_io1/parameters.yaml"

    extra_args = {}
    if not algorithm.run_args.requires_root:
        # run the container as the current user to ensure written files are always owned by the user
        # also overall better security-wise
        extra_args["user"] = f"{os.getuid()}:{os.getgid()}"

    # device setup
    device_requests = _handle_device_requests(
        algorithm=algorithm, cuda_devices=cuda_devices, force_cpu=force_cpu
    )

    # Run the container
    container = client.containers.run(
        image=algorithm.run_args.docker_image,
        volumes=volume_mappings,
        device_requests=device_requests,
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

    # TODO add option to print/ save container output

    logger.info(f"{' Finished inference ':-^80}")
