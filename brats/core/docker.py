from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import docker
import nibabel as nib
import numpy as np
from docker.errors import DockerException
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich import box

from brats.constants import DUMMY_PARAMETERS, PARAMETERS_DIR, PACKAGE_CITATION
from brats.utils.algorithm_config import AlgorithmData
from brats.utils.exceptions import (
    AlgorithmNotCPUCompatibleException,
    BraTSContainerException,
)
from brats.utils.zenodo import check_additional_files_path, get_dummy_path

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
    status = line.get("status")
    layer_id = line.get("id", "unknown")

    if status in {"Downloading", "Extracting"}:
        task_key = f"[{status} {layer_id}]"

        progress_detail = line.get("progressDetail") or {}
        total = progress_detail.get("total")
        current = progress_detail.get("current")

        # Case 1: we know total size -> normal progress bar
        if total:
            if task_key not in tasks:
                tasks[task_key] = progress.add_task(f"{task_key}", total=total)
            progress.update(tasks[task_key], completed=current or 0)

        # Case 2: no total/current -> animated bar showing indefinite progress
        else:
            if task_key not in tasks:
                tasks[task_key] = progress.add_task(
                    f"{task_key}", total=None  # total=None means indeterminate
                )
            progress.update(tasks[task_key], advance=0.1)


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
        logger.info("Forcing CPU execution")
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
    # ensure additional_files are present and get path
    if algorithm.additional_files is not None:
        return check_additional_files_path(
            record_id=algorithm.additional_files.record_id
        )
    else:
        # if no additional_files are directly specified a dummy additional_files folder will be mounted
        return get_dummy_path()


def _get_volume_mappings(
    data_path: Path,
    additional_files_path: Path,
    output_path: Path,
    parameters_path: Path,
) -> Dict:
    """Get the volume mappings for the docker container.

    Args:
        data_path (Path): The path to the input data
        additional_files_path (Path): The path to the additional files
        output_path (Path): The path to save the output
        parameters_path (Path): The path to mount for the parameters file

    Returns:
        Dict: The volume mappings
    """
    # TODO: add support for recommended "ro" mount mode for input data
    # data = mlcube_io0, additional files = mlcube_io1, output = mlcube_io2, parameters = mlcube_io3
    return {
        volume.absolute(): {
            "bind": f"/mlcube_io{i}",
            "mode": "rw",
        }
        for i, volume in enumerate(
            [data_path, additional_files_path, output_path, parameters_path]
        )
    }


def _get_parameters_arg(algorithm: AlgorithmData) -> Optional[str]:
    """Get the parameters argument for the docker container.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        Optional[str]: The parameters argument for the docker container or None if a parameter file is not required
    """
    if algorithm.run_args.parameters_file:
        # Docker image name is used as the identifier for the param file
        identifier = algorithm.run_args.docker_image.split(":")[0].split("/")[-1]
        file = PARAMETERS_DIR / f"{identifier}.yml"
        # Some algorithms do require a param file to be present but don't actually use it
        # In this case we simply use a dummy file
        param_file = file if file.exists() else DUMMY_PARAMETERS
        return f" --parameters_file=/mlcube_io3/{param_file.name}"
    return None


def _build_args(
    algorithm: AlgorithmData,
) -> Tuple[str, str]:
    """Build the command and extra arguments for the docker container.

    Args:
        algorithm (AlgorithmData): The algorithm data

    Returns:
        command_args, extra_args (Tuple): The command arguments and extra arguments
    """
    # Build command that will be run in the docker container
    command_args = f"--data_path=/mlcube_io0 --output_path=/mlcube_io2"
    if algorithm.additional_files is not None:
        for i, param in enumerate(algorithm.additional_files.param_name):
            additional_files_arg = f"--{param}=/mlcube_io1"
            if algorithm.additional_files.param_path:
                additional_files_arg += f"/{algorithm.additional_files.param_path[i]}"
            command_args += f" {additional_files_arg}"

    # Add parameters file arg if required
    params_arg = _get_parameters_arg(algorithm=algorithm)
    if params_arg:
        command_args += params_arg

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
            [line.decode("utf-8", errors="replace") for line in container_output]
        )
        # Check if the container exited with an error
        if exit_code["StatusCode"] != 0:
            logger.error(f">> {container_output}")
            raise BraTSContainerException(
                "Container finished with an error. See logs above for details."
            )

    return container_output


def _sanity_check_output(
    data_path: Path,
    output_path: Path,
    container_output: str,
    internal_external_name_map: Optional[Dict[str, str]] = None,
) -> None:
    """Sanity check that the number of output files matches the number of input files and the output is not empty.

    Args:
        data_path (Path): The path to the input data
        output_path (Path): The path to the output data
        container_output (str): The output of the docker container
        internal_external_name_map (Optional[Dict[str, str]]): Dictionary mapping internal name (in standardized format) to external subject name provided by user (only used for batch inference)

    Raises:
        BraTSContainerException: If not enough output files exist
    """

    # some algorithms create extra files in the data folder, so we only check for files starting with "BraTS"
    # (should result in only counting actual inputs)
    inputs = [e for e in data_path.iterdir() if e.name.startswith("BraTS")]
    outputs = list(output_path.iterdir())
    if len(outputs) < len(inputs):
        logger.error(f"Docker container output: \n\r{container_output}")
        raise BraTSContainerException(
            f"Not enough output files were created by the algorithm. Expected: {len(inputs)} Got: {len(outputs)}. Please check the logging output of the docker container for more information."
        )

    for i, output in enumerate(outputs, start=1):
        content = nib.load(output).get_fdata()
        if np.count_nonzero(content) == 0:
            name = ""
            if internal_external_name_map is not None:
                name_key = [
                    k
                    for k in internal_external_name_map.keys()
                    if output.name.startswith(k)
                ]
                if name_key:
                    name = internal_external_name_map[name_key[0]]

            logger.warning(
                f"""Output file for subject {name + " "}contains only zeros.
                Potentially the selected algorithm might not work properly with your data unless this behavior is correct for your use case.
                If this seems wrong please try to use one of the other provided algorithms and file an issue on GitHub if the problem persists."""
            )


def _log_algorithm_info(algorithm: AlgorithmData):
    """Log information about the algorithm and citation reminder.

    Args:
        algorithm (AlgorithmData): algorithm data
    """

    # intentional prints! (to always display them even when logging is disabled)
    console = Console()
    console.rule("[bold red]Citation Reminder[/bold red]")
    console.print(
        "Please support our development by citing the relevant manuscripts for the used algorithm:\n"
    )
    table = Table(
        show_header=False,
        show_lines=True,
        show_edge=False,
        box=box.ASCII,
    )
    table.add_column("type", justify="right", style="cyan")
    table.add_column("url", justify="left", style="white")
    table.add_row("BraTS Package", PACKAGE_CITATION)
    table.add_row(
        f"Challenge ({algorithm.meta.challenge} {algorithm.meta.year})",
        algorithm.meta.challenge_manuscript,
    )
    table.add_row(f"Algorithm ({algorithm.meta.authors})", algorithm.meta.paper)
    if algorithm.meta.dataset_manuscript:
        table.add_row(f"Dataset", algorithm.meta.dataset_manuscript)

    console.print(table)
    console.rule()

    logger.opt(colors=True).info(
        f"Running algorithm: <light-green> BraTS {algorithm.meta.year} {algorithm.meta.challenge} [{algorithm.meta.rank} place]</>"
    )
    logger.debug(f"Docker image: {algorithm.run_args.docker_image}")


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
    _ensure_image(image=algorithm.run_args.docker_image)

    additional_files_path = _get_additional_files_path(algorithm)

    # ensure output folder exists
    output_path.mkdir(parents=True, exist_ok=True)

    volume_mappings = _get_volume_mappings(
        data_path=data_path,
        additional_files_path=additional_files_path,
        output_path=output_path,
        parameters_path=PARAMETERS_DIR,
    )
    logger.debug(f"Volume mappings: {volume_mappings}")

    command_args, extra_args = _build_args(algorithm=algorithm)
    logger.debug(f"Command args: {command_args}, Extra args: {extra_args}")

    # device setup
    device_requests = _handle_device_requests(
        algorithm=algorithm, cuda_devices=cuda_devices, force_cpu=force_cpu
    )
    logger.debug(f"GPU Device requests: {device_requests}")

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
        data_path=data_path,
        output_path=output_path,
        container_output=container_output,
        internal_external_name_map=internal_external_name_map,
    )

    logger.debug(f"Docker container output: \n\r{container_output}")

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
