from brats.utils.algorithm_config import AlgorithmData
from pathlib import Path
from typing import Dict, List, Optional
from brats.core.docker import (
    _log_algorithm_info,
    _sanity_check_output,
    _get_additional_files_path,
    _get_volume_mappings_mlcube,
    _build_command_args,
    _handle_device_requests,
)
from brats.constants import PARAMETERS_DIR
from loguru import logger
import time
from spython.main import Client


def _ensure_image(image: str) -> str:
    """Ensure the Singularity image is present on the system. If not, pull it.

    Args:
        image (str): The docker image to pull

    Returns:
        str: The path to the Singularity image
    """
    image, puller = Client.pull("docker://" + image, stream=True, pull_folder="/tmp")
    if not Path(image).exists():
        logger.info(f"Pulling Singularity image {image}")
        for line in puller:
            logger.info(line)

    return image


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


def run_container(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
    cuda_devices: str,
    force_cpu: bool,
    internal_external_name_map: Optional[Dict[str, str]] = None,
):
    """Run a docker container for the provided algorithm.

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
    print(f"Additional files path: {additional_files_path}")
    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    volume_mappings = _get_volume_mappings_mlcube(
        data_path=data_path,
        additional_files_path=additional_files_path,
        output_path=output_path,
        parameters_path=PARAMETERS_DIR,
    )
    logger.debug(f"Volume mappings: {volume_mappings}")

    command_args, extra_args = _build_command_args(algorithm=algorithm)
    logger.debug(f"Command args: {command_args}, Extra args: {extra_args}")
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

    executor = Client.run(
        image,
        options=options,
        args=["infer", *command_args.split(" "), *extra_args],
        stream=True,
        bind=singularity_bindings,
    )
    for line in executor:
        logger.info(line)

    _sanity_check_output(
        data_path=data_path,
        output_path=output_path,
        container_output="",
        internal_external_name_map=internal_external_name_map,
    )

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
