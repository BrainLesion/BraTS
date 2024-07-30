from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import time

import docker
from loguru import logger
from rich.progress import Progress
from rich.console import Console

from brats.algorithm_config import AlgorithmData
from brats.exceptions import AlgorithmNotCPUCompatibleException, BraTSContainerException
from brats.weights import check_model_weights, get_dummy_weights_path
from docker.errors import DockerException

try:
    client = docker.from_env()
except DockerException as e:
    logger.error(
        f"Failed to connect to docker daemon. Please make sure docker is installed and running. Error: {e}"
    )
    # not aborting since this happens during read the docs builds. not a great solution tbf


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
                _show_docker_pull_progress(tasks=tasks, progress=progress, line=line)


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
            cause = (
                "User tried to force CPU execution but"
                if cuda_available
                else "No Cuda installation/ GPU was found and"
            )
            raise AlgorithmNotCPUCompatibleException(
                f"{cause} the chosen algorithm is not CPU-compatible. Aborting..."
            )
        # empty device requests => run on CPU
        return []
    # request gpu with chosen devices
    return [
        docker.types.DeviceRequest(device_ids=[cuda_devices], capabilities=[["gpu"]])
    ]


def _get_additional_files_path(algorithm: AlgorithmData) -> Path:
    """Get the path to the additional files for @algorithm.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        Path to the additional files
    """
    # ensure weights are present and get path
    if algorithm.weights is not None:
        return check_model_weights(record_id=algorithm.weights.record_id)
    else:
        # if no weights are directly specified a dummy weights folder will be mounted that is potentially used for parameter files etc.
        return get_dummy_weights_path()


def _get_volume_mappings(
    data_path: Path, additional_files_path: Path, output_path: Path
) -> Dict:
    """Get the volume mappings for the docker container.

    Args:
        data_path (Path): The path to the input data
        additional_files_path (Path): The path to the additional files
        output_path (Path): The path to save the output

    Returns:
        Dict: The volume mappings
    """
    # TODO: add support for recommended "ro" mount mode for input data
    # data = mlcube_io0, additional files = mlcube_io1, output = mlcube_io2
    return {
        volume.absolute(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, volume in enumerate([data_path, additional_files_path, output_path])
    }


def _build_args(
    algorithm: AlgorithmData, additional_files_path: Path
) -> Tuple[str, str]:
    """Build the command and extra arguments for the docker container.

    Args:
        algorithm (AlgorithmData): The algorithm data
        additional_files_path (Path): The path to the additional files

    Returns:
        command_args, extra_args (Tuple): The command arguments and extra arguments
    """
    # Build command that will be run in the docker container
    command_args = (
        f"--data_path=/mlcube_io0 --{algorithm.weights.param_name}=/mlcube_io1 --output_path=/mlcube_io2"
        if algorithm.weights is not None
        else f"--data_path=/mlcube_io0 --output_path=/mlcube_io2"
    )

    if algorithm.run_args.parameters_file:
        # The algorithms that need a parameters file do not seem to actually use it but just need it to exist
        # As a workaround we simply create an empty file
        parameters_file = additional_files_path / "parameters.yaml"
        parameters_file.touch()
        command_args += f" --parameters_file=/mlcube_io1/parameters.yaml"

    extra_args = {}
    if not algorithm.run_args.requires_root:
        # run the container as the current user to ensure written files are always owned by the user
        # also overall better security-wise
        extra_args["user"] = f"{os.getuid()}:{os.getgid()}"

    return command_args, extra_args


def _observe_docker_output(container: docker.models.containers.Container) -> str:
    """Observe the output of a running docker container and display a spinner. On Errors log container output.

    Args:
        container (docker.models.containers.Container): The container to observe
    """
    # capture the output
    container_output = container.attach(
        stdout=True, stderr=True, stream=True, logs=True
    )

    # Display spinner while the container is running
    with Console().status("Running inference..."):
        # Wait for the container to finish
        exit_code = container.wait()
        container_output = "\n\r".join(
            [line.decode("utf-8") for line in container_output]
        )
        # Check if the container exited with an error
        if exit_code["StatusCode"] != 0:
            logger.error(f">> {container_output}")
            raise BraTSContainerException(
                "Container finished with an error. See logs above for details."
            )

    return container_output


def _sanity_check_output(
    data_path: Path, output_path: Path, container_output: str
) -> None:
    """Sanity check that the number of output files matches the number of input files.

    Args:
        data_path (Path): The path to the input data
        output_path (Path): The path to the output data
        container_output (str): The output of the docker container

    Raises:
        BraTSContainerException: If not enough output files exist
    """

    inputs = list(data_path.iterdir())
    outputs = list(output_path.iterdir())

    if len(inputs) == len(outputs) and len(outputs) > 0:
        return

    if len(outputs) < len(inputs):
        logger.error(f"Docker container output: \n\r{container_output}")
        raise BraTSContainerException(
            f"Not enough output files were created by the algorithm. Expected: {len(inputs)} Got: {len(outputs)}. Please check the logging output of the docker container for more information."
        )


def run_docker(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
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
    logger.debug(f"Docker image: {algorithm.run_args.docker_image}")

    # ensure image is present, if not pull it
    _ensure_image(image=algorithm.run_args.docker_image)
    # Log the message
    additional_files_path = _get_additional_files_path(algorithm)

    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    volume_mappings = _get_volume_mappings(
        data_path=data_path,
        additional_files_path=additional_files_path,
        output_path=output_path,
    )
    logger.debug(f"Volume mappings: {volume_mappings}")

    command_args, extra_args = _build_args(
        algorithm=algorithm, additional_files_path=additional_files_path
    )
    logger.debug(f"Command args: {command_args}, Extra args: {extra_args}")

    # device setup
    device_requests = _handle_device_requests(
        algorithm=algorithm, cuda_devices=cuda_devices, force_cpu=force_cpu
    )
    logger.debug(f"Device requests: {device_requests}")

    # Run the container
    logger.info(f"{'Starting inference'}")
    start_time = time.time()
    container = client.containers.run(
        image=algorithm.run_args.docker_image,
        volumes=volume_mappings,
        device_requests=device_requests,
        command=f"infer {command_args}",
        network_mode="none",
        detach=True,
        remove=True,
        shm_size=algorithm.run_args.shm_size,
        **extra_args,
    )
    container_output = _observe_docker_output(container=container)
    _sanity_check_output(
        data_path=data_path, output_path=output_path, container_output=container_output
    )

    logger.debug(f"Docker container output: \n\r{container_output}")

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
