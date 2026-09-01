from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import docker
from loguru import logger
from spython.main import Client

from brats.constants import PARAMETERS_DIR
from brats.core.docker import (
    _ensure_image as _ensure_docker_image,
)
from brats.core.docker import (
    _get_additional_files_path,
    _get_parameters_arg,
    _get_volume_mappings_docker_only,
    _get_volume_mappings_mlcube,
    _handle_device_requests,
    _log_algorithm_info,
    _sanity_check_output,
)
from brats.utils.algorithm_config import AlgorithmData
from brats.utils.cuda import normalize_cuda_devices

try:
    docker_client = docker.from_env()
except docker.errors.DockerException:
    logger.debug(
        "Could not connect to the Docker daemon. Docker functionality is "
        "disabled, so the Singularity container's working directory may not "
        "be set correctly."
    )
    docker_client = None


def _build_command_args(
    algorithm: AlgorithmData,
) -> list[str]:
    """Build the command arguments for the singularity container.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        List[str]: The command arguments
    """

    command_args = ["--data_path=/mlcube_io0", "--output_path=/mlcube_io2"]
    if algorithm.additional_files is not None:
        param_name = algorithm.additional_files.param_name or []
        for i, param in enumerate(param_name):
            additional_files_arg = f"--{param}=/mlcube_io1"
            if algorithm.additional_files.param_path:
                additional_files_arg += f"/{algorithm.additional_files.param_path[i]}"
            command_args.append(additional_files_arg)

    # Add parameters file arg if required
    params_arg = _get_parameters_arg(algorithm=algorithm)
    if params_arg:
        command_args.append(params_arg.strip())

    return command_args


def _ensure_image(image: str) -> str:
    """
    Ensure the Singularity image is present on the system. If not, pull it
    as a Sandbox.
    This function checks if the specified Singularity image exists locally
    in the temporary directory.
    If the image is not found, it pulls the image from Docker Hub, creates
    a Singularity Sandbox at the target location.

    Args:
        image (str): The Docker image to pull and convert into a Singularity Sandbox.

    Returns:
        str: The path to the Singularity image Sandbox.
    """
    persistent_dir = os.path.join(tempfile.gettempdir(), "brats_singularity_images")
    os.makedirs(persistent_dir, exist_ok=True)
    logger.debug(f"Persistent folder: {persistent_dir}")
    temp_folder = Path(persistent_dir)
    image_path = temp_folder.joinpath(image.replace(":", "_"))
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Pulling Singularity image {image} and creating a Sandbox at {image_path}"
        )
        subprocess.run(
            [
                "singularity",
                "build",
                "--sandbox",
                "--fakeroot",
                str(image_path),
                f"docker://{image}",
            ],
            check=True,
        )

    return str(image_path)


def _convert_volume_mappings_to_singularity_format(
    volume_mappings: dict[Any, dict[str, str]],
) -> list[str]:
    """Convert volume mappings from Docker format to Singularity format.

    Args:
        volume_mappings (Dict[Path | str, Dict[str, str]]): The volume mappings in Docker format

    Returns:
        List[str]: The volume mappings in Singularity format
    """
    singularity_bindings = []
    for host_path, val in volume_mappings.items():
        container_path = val["bind"]
        singularity_bindings.append(f"{host_path}:{container_path}")
    return singularity_bindings


def _get_docker_working_dir(image: str) -> Optional[Path]:
    """
    Retrieve the working directory configured in the Docker image.

    This is required to properly initialize the working directory for the
    Singularity container, ensuring that the container starts in the correct
    location as defined by the Docker image.

    Args:
        image (str): The Docker image name or ID.

    Returns:
        Path | None: The working directory specified in the Docker image
            configuration. None if docker client is not available.
    """
    if docker_client is None:
        return None
    try:
        logger.debug(f"Inspecting image {image}")
        image_obj = docker_client.images.get(image)
    except docker.errors.ImageNotFound:
        logger.debug(f"Image {image} not found locally.")
        _ensure_docker_image(image)
        image_obj = docker_client.images.get(image)
    workdir = image_obj.attrs["Config"].get("WorkingDir", None)
    logger.debug(f"Working directory: {workdir}")
    if workdir is None:
        return None
    else:
        return Path(workdir)


_SINGULARITY_CUDA_ENV_VAR = "SINGULARITYENV_CUDA_VISIBLE_DEVICES"
_HOST_CUDA_ENV_VAR = "CUDA_VISIBLE_DEVICES"


@contextmanager
def _cuda_device_selection_env(cuda_devices: str, enabled: bool) -> Iterator[None]:
    """Temporarily expose only the requested GPUs inside the container.

    Singularity's ``--nv`` flag exposes all host GPUs, so GPU selection is
    enforced via the ``SINGULARITYENV_CUDA_VISIBLE_DEVICES`` environment
    variable, which Singularity passes into the container as
    ``CUDA_VISIBLE_DEVICES``. The previous value of the variable (if any) is
    restored on exit.

    Args:
        cuda_devices (str): Comma-separated list of host GPU IDs to make
            visible inside the container.
        enabled (bool): Whether to restrict GPUs at all. If ``False`` (CPU
            run), the environment is left untouched.

    Note:
        The environment override is process-global and therefore not
        thread-safe. Concurrent Singularity runs within the same process
        may interfere with each other's GPU selection; run containers
        sequentially.
    """
    if not enabled:
        yield
        return

    previous_value = os.environ.get(_SINGULARITY_CUDA_ENV_VAR)
    if previous_value is not None:
        logger.warning(
            f"The {_SINGULARITY_CUDA_ENV_VAR} environment variable is already "
            f"set to '{previous_value}'. It will be overridden with the "
            f"cuda_devices parameter value '{cuda_devices}'."
        )
    if os.environ.get(_HOST_CUDA_ENV_VAR) is not None:
        logger.warning(
            f"The host's {_HOST_CUDA_ENV_VAR} environment variable is set "
            f"(e.g. by a batch scheduler). The cuda_devices parameter refers "
            f"to host GPU IDs and takes precedence inside the container."
        )
    os.environ[_SINGULARITY_CUDA_ENV_VAR] = normalize_cuda_devices(cuda_devices)
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(_SINGULARITY_CUDA_ENV_VAR, None)
        else:
            os.environ[_SINGULARITY_CUDA_ENV_VAR] = previous_value


def run_container(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
    cuda_devices: str,
    force_cpu: bool,
    internal_external_name_map: Optional[dict[str, str]] = None,
    overlay_size: int = 1024,
) -> None:
    """Run a Singularity container for the provided algorithm.

    Args:
        algorithm (AlgorithmData): The data of the algorithm to run
        data_path (Path | str): The path to the input data
        output_path (Path | str): The path to save the output
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
        internal_external_name_map (Dict[str, str]): Dictionary mapping
            internal name (in standardized format) to external subject name
            provided by user (only used for batch inference)
        overlay_size (int): The size of the overlay image in MB. Defaults to 1024.
    """
    if overlay_size <= 0:
        raise ValueError("Overlay size must be greater than 0.")
    _log_algorithm_info(algorithm=algorithm)
    # ensure image is present, if not pull it
    image = _ensure_image(image=algorithm.run_args.docker_image)

    additional_files_path = _get_additional_files_path(algorithm)
    logger.debug(f"Additional files path: {additional_files_path}")
    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    command_args = _build_command_args(algorithm=algorithm)
    command_args_str = " ".join(command_args)
    logger.debug(f"Command args: {command_args_str}")

    volume_mappings: dict[Any, dict[str, str]]
    if algorithm.meta.year <= 2024:
        volume_mappings = _get_volume_mappings_mlcube(
            data_path=data_path,
            additional_files_path=additional_files_path,
            output_path=output_path,
            parameters_path=PARAMETERS_DIR,
        )
        args = ["infer", *command_args]
    else:
        volume_mappings = _get_volume_mappings_docker_only(
            data_path=data_path,
            output_path=output_path,
        )
        args = None

    logger.debug(f"Volume mappings: {volume_mappings}")

    # device setup
    device_requests = _handle_device_requests(
        algorithm=algorithm, cuda_devices=cuda_devices, force_cpu=force_cpu
    )
    logger.debug(f"GPU Device requests: {device_requests}")

    # Run the container
    logger.info("Starting inference")
    start_time = time.time()

    singularity_bindings = _convert_volume_mappings_to_singularity_format(
        volume_mappings
    )

    options = []

    enable_gpu = len(device_requests) > 0 and not force_cpu
    if enable_gpu:
        logger.info(
            f"Restricting container GPUs to CUDA devices: {cuda_devices} "
            f"(via {_SINGULARITY_CUDA_ENV_VAR})"
        )
        options.append("--nv")  # Singularity uses --nv to enable GPU support

    # TODO: The --fakeroot option may be required for certain algorithms that
    # need root privileges inside the Singularity container.
    docker_working_dir = _get_docker_working_dir(algorithm.run_args.docker_image)
    if docker_working_dir is not None:
        options.append("--cwd")
        options.append(str(docker_working_dir))
    else:
        logger.warning(
            "Docker working directory not found. Using default working directory."
        )
    overlay_path = Path(image).parent / (Path(image).name + "_overlay.img")
    options.append("--overlay")
    options.append(str(overlay_path))

    overlay_created = False
    if not overlay_path.exists():
        subprocess.run(
            [
                "singularity",
                "overlay",
                "create",
                "--size",
                str(overlay_size),
                str(overlay_path),
            ],
            check=True,
        )
        overlay_created = True
    try:
        with _cuda_device_selection_env(cuda_devices=cuda_devices, enabled=enable_gpu):
            executor = Client.run(
                image,
                options=options,
                args=args,
                stream=True,
                bind=singularity_bindings,
            )
            container_output = list(executor)

        _sanity_check_output(
            data_path=data_path,
            output_path=output_path,
            container_output="\n".join(container_output),
            internal_external_name_map=internal_external_name_map,
        )
    finally:
        try:
            if overlay_created:
                overlay_path.unlink()
        except FileNotFoundError:
            pass
    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
