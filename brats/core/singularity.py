from brats.utils.algorithm_config import AlgorithmData
from pathlib import Path

import subprocess
from typing import Dict, List, Optional
from brats.core.docker import (
    _log_algorithm_info,
    _sanity_check_output,
    _get_additional_files_path,
    _get_volume_mappings_mlcube,
    _get_parameters_arg,
    _handle_device_requests,
    _get_volume_mappings_docker_only,
    _ensure_image as _ensure_docker_image,
)
from brats.constants import PARAMETERS_DIR
from loguru import logger
import time
from spython.main import Client
import docker

try:
    client = docker.from_env()
except docker.errors.DockerException as e:
    logger.error(
        f"Failed to connect to docker daemon. Please make sure docker is installed and running. Error: {e}"
    )


def _build_command_args(
    algorithm: AlgorithmData,
) -> str:
    """Build the command arguments for the singularity container.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        command_args: The command arguments
    """

    command_args = ["--data_path=/mlcube_io0", "--output_path=/mlcube_io2"]
    if algorithm.additional_files is not None:
        for i, param in enumerate(algorithm.additional_files.param_name):
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
    Ensure the Singularity image is present on the system. If not, pull it as a Sandbox and create an overlay image.
    This function checks if the specified Singularity image exists locally in the /tmp directory.
    If the image is not found, it pulls the image from Docker Hub, creates a Singularity Sandbox at the target location,
    and also creates an overlay image. The overlay image allows the Singularity run command
    to write inside the image sandbox, enabling writable operations during container execution.

    Args:
        image (str): The Docker image to pull and convert into a Singularity Sandbox.

    Returns:
        str: The path to the Singularity image Sandbox.
    """
    image_path = Path("/tmp").joinpath(image.split(":")[0])
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
        # puller = Client.build("docker://" + image, str(image_path), sandbox=True, options=["--fakeroot"])
        # for line in puller:
        #   logger.debug(line)

    return str(image_path)


def _convert_volume_mappings_to_singularity_format(
    volume_mappings: Dict[Path, Path],
) -> List[str]:
    """Convert volume mappings from Docker format to Singularity format.

    Args:
        volume_mappings (Dict[Path, Path]): The volume mappings in Docker format

    Returns:
        List[str]: The volume mappings in Singularity format
    """
    singularity_bindings = []
    for host_path, val in volume_mappings.items():
        container_path = val["bind"]
        singularity_bindings.append(f"{str(host_path)}:{container_path}")
    return singularity_bindings


def _get_docker_working_dir(image: str) -> Path:
    """
    Retrieve the working directory configured in the Docker image.

    This is required to properly initialize the working directory for the Singularity container,
    ensuring that the container starts in the correct location as defined by the Docker image.

    Args:
        image (str): The Docker image name or ID.

    Returns:
        Path: The working directory specified in the Docker image configuration.
    """
    _ensure_docker_image(image)
    logger.debug(f"Inspecting image {image}")
    image = client.images.get(image)
    workdir = image.attrs["Config"].get("WorkingDir", "")
    logger.debug(f"Working directory: {workdir}")
    return Path(workdir)


def run_container(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
    cuda_devices: str,
    force_cpu: bool,
    internal_external_name_map: Optional[Dict[str, str]] = None,
):
    """Run a Singularity container for the provided algorithm.

    Args:
        algorithm (AlgorithmData): The data of the algorithm to run
        data_path (Path | str): The path to the input data
        output_path (Path | str): The path to save the output
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
        internal_external_name_map (Dict[str, str]): Dictionary mapping internal name (in standardized format) to external subject name provided by user (only used for batch inference)
    """
    _log_algorithm_info(algorithm=algorithm)
    # ensure image is present, if not pull it
    image = _ensure_image(image=algorithm.run_args.docker_image)

    additional_files_path = _get_additional_files_path(algorithm)
    logger.debug(f"Additional files path: {additional_files_path}")
    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    command_args = _build_command_args(algorithm=algorithm)
    logger.debug(f"Command args: {" ".join(command_args)}")
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
    logger.info(f"{'Starting inference'}")
    start_time = time.time()

    singularity_bindings = _convert_volume_mappings_to_singularity_format(
        volume_mappings
    )

    options = []

    if len(device_requests) > 0 and not force_cpu:
        logger.info(f"Using CUDA devices: {cuda_devices}")
        options.append("--nv")  # Singularity uses --nv to enable GPU support

    # options.append(f"--fakeroot")
    options.append("--cwd")
    options.append(str(_get_docker_working_dir(algorithm.run_args.docker_image)))
    options.append("--overlay")
    options.append(str(image) + "_overlay.img")

    subprocess.run(
        [
            "singularity",
            "overlay",
            "create",
            "--size",
            "1024",
            str(image) + "_overlay.img",
        ],
        check=True,
    )
    executor = Client.run(
        image,
        options=options,
        args=args,
        stream=True,
        bind=singularity_bindings,
    )
    container_output = []
    for line in executor:
        container_output.append(line)

    _sanity_check_output(
        data_path=data_path,
        output_path=output_path,
        container_output="\n".join(container_output),
        internal_external_name_map=internal_external_name_map,
    )
    try:
        Path(str(image) + "_overlay.img").unlink()
    except FileNotFoundError:
        pass
    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
