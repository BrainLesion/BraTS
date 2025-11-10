from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional
import random
import string
import base64
from loguru import logger
from kubernetes import client, config
from kubernetes.stream import stream
from brats.constants import PARAMETERS_DIR
from brats.utils.algorithm_config import AlgorithmData
from brats.core.docker import (
    _log_algorithm_info,
    _get_container_user,
    _handle_device_requests,
    _get_parameters_arg,
    _sanity_check_output,
)
import io
import tarfile
from typing import List
from brats.utils.zenodo import (
    _get_zenodo_metadata_and_archive_url,
    ZenodoException,
    get_dummy_path,
)


def _build_command_args(
    algorithm: AlgorithmData,
    additional_files_path: str,
    data_path: str,
    output_path: str,
    mount_path: str = "/data",
) -> List[str]:
    """Build the command arguments for the Kubernetes job.

    Args:
        algorithm (AlgorithmData): The algorithm data
        additional_files_path (str): The path to the additional files
        data_path (str): The path to the input data
        output_path (str): The path to save the output
        mount_path (str): The path to mount the PVC to. Defaults to "/data".
    Returns:
        command_args: The command arguments
    """
    command_args = f"--data_path={str(data_path)} --output_path={str(output_path)}"
    if algorithm.additional_files is not None:
        for i, param in enumerate(algorithm.additional_files.param_name):
            additional_files_arg = f"--{param}={str(additional_files_path)}"
            if algorithm.additional_files.param_path:
                additional_files_arg += f"/{algorithm.additional_files.param_path[i]}"
            command_args += f" {additional_files_arg}"

    params_arg = _get_parameters_arg(algorithm=algorithm)
    if params_arg:
        command_args += params_arg.replace(
            "/mlcube_io3", str(Path(mount_path).joinpath("parameters"))
        )

    return command_args


def _observe_job_output(pod_name: str, namespace: str) -> str:
    """Observe the output of a running job.
    Args:
        pod_name (str): The name of the pod to observe the output of
        namespace (str): The namespace of the pod to observe the output of
    """
    v1 = client.CoreV1Api()

    for _ in range(300):  # up to 10 minutes
        pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
        pod_phase = pod.status.phase
        statuses = pod.status.container_statuses
        is_running = False
        if statuses:
            for status in statuses:
                if (
                    status.name == "job-container"
                    and status.state
                    and status.state.running
                ):
                    is_running = True
                    break
        if pod_phase == "Running" and is_running:
            break
        elif pod_phase in ["Failed", "Succeeded"]:
            logger.warning(f"Pod '{pod_name}' entered terminal phase: {pod_phase}")
            break
        time.sleep(2)
    else:
        logger.error(
            f"Timed out waiting for main container in pod '{pod_name}' to be running"
        )
        return ""
    try:
        log = v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container="job-container",
            follow=True,
            _preload_content=True,
        )
        return log
    except Exception as e:
        logger.error(
            f"Failed to fetch logs from pod '{pod_name}' in namespace '{namespace}': {e}"
        )
        return ""


def _execute_command_in_pod(
    pod_name: str,
    namespace: str,
    command: List[str],
    container: str,
    stderr: bool = True,
    stdin: bool = False,
    stdout: bool = True,
    tty: bool = False,
    _preload_content: bool = True,
) -> str:
    """Execute a command in a pod.
    Args:
        pod_name (str): The name of the pod to execute the command in
        namespace (str): The namespace of the pod to execute the command in
        command (List[str]): The command to execute
        container (str): The container to execute the command in
        stderr (bool): Whether to capture stderr. Defaults to True.
        stdin (bool): Whether to capture stdin. Defaults to False.
        stdout (bool): Whether to capture stdout. Defaults to True.
        tty (bool): Whether to use a TTY. Defaults to False.
        _preload_content (bool): Whether to preload the content. Defaults to True.
    Returns:
        str: The output of the command
    """
    v1 = client.CoreV1Api()
    output = stream(
        v1.connect_get_namespaced_pod_exec,
        pod_name,
        namespace,
        command=command,
        container=container,
        stderr=stderr,
        stdin=stdin,
        stdout=stdout,
        tty=tty,
        _preload_content=_preload_content,
    )
    logger.debug(
        f"Command '{command}' executed successfully in pod '{pod_name}' in namespace '{namespace}'.\nOutput:\n{output}"
    )
    return output


def _download_additional_files(
    algorithm: AlgorithmData, pod_name: str, namespace: str, mount_path: str = "/data"
) -> Path:
    """Download additional files from Zenodo.
    Args:
        algorithm (AlgorithmData): The algorithm data
        pod_name (str): The name of the pod to download the additional files to
        namespace (str): The namespace of the pod to download the additional files to
        mount_path (str): The path to mount the PVC to. Defaults to "/data".
    """
    if algorithm.additional_files is not None:
        zenodo_response = _get_zenodo_metadata_and_archive_url(
            record_id=algorithm.additional_files.record_id
        )
        if not zenodo_response:
            msg = "Additional files not found locally and Zenodo could not be reached. Exiting..."
            logger.error(msg)
            raise ZenodoException(msg)

        zenodo_metadata, archive_url = zenodo_response
        record_folder = str(
            Path(mount_path).joinpath(
                f"{algorithm.additional_files.record_id}_v{zenodo_metadata['version']}"
            )
        )

        commands = [
            "sh",
            "-c",
            (
                f'if [ ! -d {record_folder} ] || [ -z "$(ls -A {record_folder})" ]; then '
                f"  mkdir -p {record_folder} && "
                f"  wget -O {record_folder}/archive.zip {archive_url} && "
                f"  apk add --no-cache unzip && "
                f"  unzip {record_folder}/archive.zip -d {record_folder} && "
                f"  rm {record_folder}/archive.zip && "
                f"  for f in {record_folder}/*.zip; do "
                f'    if [ -f "$f" ]; then unzip "$f" -d {record_folder} && rm "$f"; fi; '
                f"  done "
                f"else "
                f"  echo 'Additional files already present in {record_folder}, skipping download.'; "
                f"fi"
            ),
        ]
        logger.info(f"Downloading additional files to {record_folder}...")
        output = _execute_command_in_pod(
            pod_name=pod_name,
            namespace=namespace,
            command=commands,
            container="init-container",
        )
        logger.info(
            f"Additional files downloaded successfully to pod '{pod_name}' in namespace '{namespace}'."
        )
        logger.info(f"Contents of {record_folder}:\n{output}")
        return Path(record_folder)
    else:
        return get_dummy_path()


def _check_files_in_pod(
    pod_name: str, namespace: str, paths: List[Path], mount_path: str = "/data"
) -> None:
    """Check if all the local files are present in the mounted path inside the pod.
    Args:
        pod_name (str): Name of the pod to check files in
        namespace (str): The Kubernetes namespace to check files in
        paths (List[Path]): List of local files to check
        mount_path (str): The path to mount the PVC to. Defaults to "/data".
    """
    logger.debug(
        f"Checking files in pod '{pod_name}' in namespace '{namespace}' with mount path '{mount_path}'."
    )
    for path in paths:
        for file in path.glob("**/*"):
            if file.is_file():
                commands = [
                    "ls",
                    "-la",
                    str(Path(mount_path).joinpath("input", file.relative_to(path))),
                ]
                output = _execute_command_in_pod(
                    pod_name=pod_name,
                    namespace=namespace,
                    command=commands,
                    container="init-container",
                )
                if "No such file or directory" in output:
                    logger.warning(
                        f"File '{file.relative_to(path)}' is not present in pod '{pod_name}' in namespace '{namespace}'. Uploading it now..."
                    )
                    _upload_files_to_pod(
                        pod_name=pod_name,
                        namespace=namespace,
                        paths=[file],
                        mount_path=mount_path,
                        relative_to=path,
                        parent_dir="input",
                    )


def _download_folder_from_pod(
    pod_name: str,
    namespace: str,
    container: str,
    remote_paths: List[Path],
    local_base_dir: Path = Path(".").absolute(),
):
    """Download a folder from a pod to a local directory.
    Args:
        pod_name (str): The name of the pod to download the folder from
        namespace (str): The namespace of the pod to download the folder from
        container (str): The container to download the folder from
        remote_paths (List[Path]): The paths to the remote folder to download
        local_base_dir (Path): The base directory to download the folder to. Defaults to the current directory.
    """

    for path in remote_paths:
        folder_name = str(path)

        command = ["sh", "-c", f"tar cf - -C {folder_name} . | base64"]
        resp = _execute_command_in_pod(
            pod_name=pod_name,
            namespace=namespace,
            command=command,
            container=container,
            stderr=True,
            stdin=False,
            stdout=True,
            _preload_content=False,
        )

        base64_chunks = []

        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunk = resp.read_stdout()
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                base64_chunks.append(chunk)
            if resp.peek_stderr():
                err = resp.read_stderr()
                if err:
                    logger.error(f"STDERR: {err}")
        resp.close()

        full_base64 = b"".join(base64_chunks)
        tar_data = base64.b64decode(full_base64)

        base_folder_name = Path(folder_name).name
        tarfile_path = local_base_dir / f"{base_folder_name}"
        with open(tarfile_path, "wb") as tarfile_obj:
            tarfile_obj.write(tar_data)

        with tarfile.open(tarfile_path, "r") as tar:
            tar.extractall(path=local_base_dir)

        tarfile_path.unlink()


def _upload_files_to_pod(
    pod_name: str,
    namespace: str,
    paths: List[Path],
    mount_path: str = "/data",
    relative_to: Path = None,
    parent_dir: Path = None,
) -> None:
    """Upload files to a pod in the specified namespace.
    Args:
        pod_name (str): Name of the pod to upload files to
        namespace (str): The Kubernetes namespace to upload files to
        paths (List[Path]): List of local files or directories to upload
        mount_path (str): The path to mount the PVC to. Defaults to "/data".
        relative_to (Path): The path to relativize the files to. Defaults to None.
        parent_dir (Path): The parent directory of the files to upload. Defaults to None.
    """

    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        for path in paths:
            if path.is_file():
                tar.add(
                    path,
                    arcname=(
                        Path(parent_dir).joinpath(path.relative_to(relative_to))
                        if parent_dir
                        else path.relative_to(relative_to)
                    ),
                )
            else:
                for file in path.glob("**/*"):
                    if file.is_file():
                        tar.add(
                            file,
                            arcname=(
                                Path(parent_dir).joinpath(file.relative_to(path))
                                if parent_dir
                                else file.relative_to(path)
                            ),
                        )
    tar_stream.seek(0)
    commands = ["tar", "xmf", "-", "-C", mount_path]
    resp = _execute_command_in_pod(
        pod_name=pod_name,
        namespace=namespace,
        command=commands,
        container="init-container",
        stdin=True,
        _preload_content=False,
    )

    resp.write_stdin(tar_stream.read())
    resp.close()
    logger.info(
        f"File uploaded successfully to pod '{pod_name}' in namespace '{namespace}'."
    )


def _create_namespaced_pvc(
    pvc_name: str, namespace: str, storage_size: str = "1Gi", storage_class: str = None
) -> None:
    """Create a namespaced PersistentVolumeClaim (PVC) in the specified namespace. If the PVC already exists, it will be skipped.

    Args:
        pvc_name (str): Name of the PVC to create
        namespace (str): The Kubernetes namespace to create the PVC in
        storage_size (str): The size of the storage to request
        storage_class (str): The storage class to use for the PVC. If None, the default storage class will be used.
    """
    core_v1_api = client.CoreV1Api()
    pvc_list = core_v1_api.list_namespaced_persistent_volume_claim(namespace=namespace)
    if len(pvc_list.items) > 0:
        for pvc in pvc_list.items:
            if pvc.metadata.name == pvc_name:
                logger.debug(
                    f"PVC '{pvc_name}' already exists in namespace '{namespace}'. Skipping creation."
                )
                return

    if storage_class is None:
        pvc_spec = client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(requests={"storage": storage_size}),
        )
    else:
        pvc_spec = client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(requests={"storage": storage_size}),
            storage_class_name=storage_class,
        )

    core_v1_api.create_namespaced_persistent_volume_claim(
        namespace=namespace,
        body=client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(name=pvc_name), spec=pvc_spec
        ),
    )


def _create_finalizer_job(
    job_name: str, namespace: str, pvc_name: str, mount_path: str = "/data"
) -> None:
    """Create a finalizer job in the specified namespace.
    Args:
        job_name (str): Name of the Job to create
        namespace (str): The Kubernetes namespace to create the Job in
        pvc_name (str): Name of the PersistentVolumeClaim (PVC) to use for this Job
        mount_path (str): The path to mount the PVC to. Defaults to "/data".
    """
    batch_v1_api = client.BatchV1Api()
    job_list = batch_v1_api.list_namespaced_job(namespace=namespace)
    if len(job_list.items) > 0:
        for job in job_list.items:
            if job.metadata.name == job_name:
                logger.warning(
                    f"Job '{job_name}' already exists in namespace '{namespace}'. Deleting it."
                )
                batch_v1_api.delete_namespaced_job(name=job_name, namespace=namespace)

    batch_v1_api.create_namespaced_job(
        namespace=namespace,
        body=client.V1Job(
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        volumes=[
                            client.V1Volume(
                                name=pvc_name,
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                    claim_name=pvc_name
                                ),
                            )
                        ],
                        containers=[
                            client.V1Container(
                                name="finalizer-container",
                                image="alpine:latest",
                                command=[
                                    "sh",
                                    "-c",
                                    "while [ ! -f /etc/content_verified ]; do sleep 1; done",
                                ],
                                volume_mounts=[
                                    client.V1VolumeMount(
                                        name=pvc_name, mount_path=mount_path
                                    )
                                ],
                            )
                        ],
                    )
                )
            ),
        ),
    )

    core_v1_api = client.CoreV1Api()
    label_selector = f"job-name={job_name}"
    pod_name = None
    for _ in range(60):
        pod_list = core_v1_api.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        )
        if pod_list.items:
            # If more than one pod, pick the first (common for most K8s jobs)
            # Pick the latest created pod (by creation timestamp)
            latest_pod = max(
                pod_list.items, key=lambda pod: pod.metadata.creation_timestamp
            )
            pod_name = latest_pod.metadata.name
            break
        time.sleep(2)
    if pod_name is None:
        raise RuntimeError(
            f"Timed out waiting for pod to be created for job {job_name}"
        )
    return pod_name


def _create_namespaced_job(
    job_name: str,
    namespace: str,
    pvc_name: str,
    image: str,
    device_requests: List[client.V1DeviceRequest],
    pv_mounts: Dict[str, str],
    args: List[str] = None,
    shm_size: str = None,
    user: str = None,
) -> None:
    """Create a namespaced Job in the specified namespace.

    Args:
        job_name (str): Name of the Job to create
        namespace (str): The Kubernetes namespace to create the Job in
        pvc_name (str): Name of the PersistentVolumeClaim (PVC) to use for this Job
        image (str): The image to use for the Job
        device_requests (List[client.V1DeviceRequest]): The device requests to use for the Job
        pv_mounts (Dict[str, str]): The PersistentVolumeClaims (PVCs) to mount to the Job.
        args (List[str]): The arguments to use for the Job. Defaults to None.
        shm_size (str): The size of the shared memory to use for the Job. Defaults to None.
        user (str): The user to run the Job as. Defaults to None (root is used if not specified).
    """
    batch_v1_api = client.BatchV1Api()
    job_list = batch_v1_api.list_namespaced_job(namespace=namespace)
    if len(job_list.items) > 0:
        for job in job_list.items:
            if job.metadata.name == job_name:
                logger.warning(
                    f"Job '{job_name}' already exists in namespace '{namespace}'. Deleting it."
                )
                batch_v1_api.delete_namespaced_job(name=job_name, namespace=namespace)
    core_v1_api = client.CoreV1Api()
    label_selector = f"job-name={job_name}"
    pod_list = core_v1_api.list_namespaced_pod(
        namespace=namespace, label_selector=label_selector
    )
    for pod in pod_list.items:
        pod_name_to_delete = pod.metadata.name
        logger.warning(
            f"Deleting pod '{pod_name_to_delete}' in namespace '{namespace}' associated with job '{job_name}'."
        )
        try:
            core_v1_api.delete_namespaced_pod(
                name=pod_name_to_delete, namespace=namespace
            )
        except Exception as e:
            logger.error(
                f"Failed to delete pod '{pod_name_to_delete}' in namespace '{namespace}': {e}"
            )

    # user_id = int(user.split(":")[0]) if user else 0  # TODO: Implement security_context for container if/when user/group IDs are required.
    # group_id = int(user.split(":")[1]) if user else 0  # TODO: Implement security_context for container if/when user/group IDs are required.
    volume_mounts = []
    for pvc_mount_name, pvc_mount_path in pv_mounts.items():
        volume_mounts.append(client.V1VolumeMount(name=pvc_mount_name, mount_path=pvc_mount_path))
    container_spec = client.V1Container(
        name="job-container",
        image=image,
        volume_mounts=volume_mounts,
        # security_context=client.V1SecurityContext(run_as_user=user_id, run_as_group=group_id)
        # TODO: Implement security_context for container if/when user/group IDs are required.
    )
    if len(device_requests) > 0:
        gpu_count = len(device_requests)
        container_spec.resources = client.V1ResourceRequirements(
            requests={"nvidia.com/gpu": gpu_count}, limits={"nvidia.com/gpu": gpu_count}
        )
    volumes = [
        client.V1Volume(
            name=pvc_name,
            persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                claim_name=pvc
            ),
        )
        for pvc in pv_mounts.keys()
    ]
    if shm_size is not None:
        shm_size_formatted = shm_size.replace("gb", "Gi")
        volumes.append(
            client.V1Volume(
                name="shm",
                empty_dir=client.V1EmptyDirVolumeSource(
                    medium="Memory", size_limit=shm_size_formatted
                ),
            )
        )
        container_spec.volume_mounts.append(
            client.V1VolumeMount(name="shm", mount_path="/dev/shm")
        )

    if args is not None:
        container_spec.args = args
    batch_v1_api.create_namespaced_job(
        namespace=namespace,
        body=client.V1Job(
            metadata=client.V1ObjectMeta(name=job_name),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        restart_policy="Never",
                        volumes=volumes,
                        init_containers=[
                            client.V1Container(
                                name="init-container",
                                image="alpine:latest",
                                command=[
                                    "sh",
                                    "-c",
                                    "while [ ! -f /etc/content_verified ]; do sleep 1; done",
                                ],
                                volume_mounts=volume_mounts,
                            )
                        ],
                        containers=[container_spec],
                    )
                )
            ),
        ),
    )

    core_v1_api = client.CoreV1Api()
    label_selector = f"job-name={job_name}"
    pod_name = None
    for _ in range(60):
        pod_list = core_v1_api.list_namespaced_pod(
            namespace=namespace, label_selector=label_selector
        )
        if pod_list.items:
            latest_pod = max(
                pod_list.items, key=lambda pod: pod.metadata.creation_timestamp
            )
            pod_name = latest_pod.metadata.name
            break
        time.sleep(2)
    if pod_name is None:
        raise RuntimeError(
            f"Timed out waiting for pod to be created for job {job_name}"
        )
    return pod_name


def run_job(
    algorithm: AlgorithmData,
    data_path: Path,
    output_path: Path,
    cuda_devices: str,
    force_cpu: bool,
    internal_external_name_map: Optional[Dict[str, str]] = None,
    namespace: Optional[str] = "default",
    pvc_name: Optional[str] = None,
    pvc_storage_size: Optional[str] = "1Gi",
    pvc_storage_class: Optional[str] = None,
    job_name: Optional[str] = None,
    data_mount_path: Optional[str] = "/data",
):
    """Run a Kubernetes job for the provided algorithm.

    Args:
        algorithm (AlgorithmData): The data of the algorithm to run
        data_path (Path | str): The path to the input data
        output_path (Path | str): The path to save the output
        cuda_devices (str): The CUDA devices to use
        force_cpu (bool): Whether to force CPU execution
        internal_external_name_map (Dict[str, str]): Dictionary mapping internal name (in standardized format) to external subject name provided by user (only used for batch inference)
        namespace (Optional[str], optional): The Kubernetes namespace to run the job in. Defaults to "default".
        pvc_name (str): Name of the PersistentVolumeClaim (PVC) to use for this job. If the PVC does not already exist, it will be created; otherwise, it must already contain the input data required for running the algorithm.
        pvc_storage_size (str): The size of the storage to request for the PVC. Defaults to "1Gi".
        pvc_storage_class (str): The storage class to use for the PVC. If None, the default storage class will be used.
        job_name (str): Name of the Job to create. If None, a random name will be generated.
        data_mount_path (str): The path to mount the PVC to. Defaults to "/data".
    """
    if pvc_name is None:
        pvc_name = (
            "brats-"
            + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
            + "-pvc"
        )
    if job_name is None:
        job_name = (
            "brats-"
            + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
            + "-job"
        )

    logger.debug(f"Job name: {job_name}")
    logger.debug(f"PersistentVolumeClaim name: {pvc_name}")
    config.load_kube_config()
    _create_namespaced_pvc(
        pvc_name=pvc_name,
        namespace=namespace,
        storage_size=pvc_storage_size,
        storage_class=pvc_storage_class,
    )
    _log_algorithm_info(algorithm=algorithm)

    device_requests = _handle_device_requests(
        algorithm=algorithm,
        cuda_devices=cuda_devices,
        force_cpu=force_cpu,
    )
    logger.debug(f"GPU Device requests: {device_requests}")
    user = _get_container_user(algorithm=algorithm)
    logger.debug(f"Container user: {user if user else 'root (required by algorithm)'}")

    if algorithm.meta.year > 2024:
        input_mount_path = "/input"
        output_mount_path = "/output"
    else:
        output_mount_path = Path(data_mount_path).joinpath("output")

    if algorithm.meta.year <= 2024:
        if algorithm.additional_files is not None:
            zenodo_response = _get_zenodo_metadata_and_archive_url(
                record_id=algorithm.additional_files.record_id
            )
            if not zenodo_response:
                msg = "Additional files not found locally and Zenodo could not be reached. Exiting..."
                logger.error(msg)
                raise ZenodoException(msg)

            zenodo_metadata, archive_url = zenodo_response
            additional_files_path = Path(data_mount_path).joinpath(
                f"{algorithm.additional_files.record_id}_v{zenodo_metadata['version']}"
            )
        else:
            additional_files_path = get_dummy_path()

        command_args = _build_command_args(
            algorithm=algorithm,
            additional_files_path=additional_files_path,
            data_path=Path(data_mount_path).joinpath("input"),
            output_path=output_mount_path,
            mount_path=data_mount_path,
        )
        command = ["infer", *command_args.split(" ")]
        pv_mounts = {
            pvc_name: data_mount_path,
        }
    else:
        command = None
        _create_namespaced_pvc(
            pvc_name=pvc_name + "-output",
            namespace=namespace,
            storage_size=pvc_storage_size,
            storage_class=pvc_storage_class,
        )

        pv_mounts = {
            pvc_name: input_mount_path,
            pvc_name + "-output": output_mount_path,
        }

    pod_name = _create_namespaced_job(
        job_name=job_name,
        namespace=namespace,
        pvc_name=pvc_name,
        pv_mounts=pv_mounts,
        image=algorithm.run_args.docker_image,
        device_requests=device_requests,
        args=command,
        shm_size=algorithm.run_args.shm_size,
        user=user,
    )
    logger.debug(f"Pod name: {pod_name}")

    core_v1_api = client.CoreV1Api()
    logger.info(f"Waiting for Pod '{pod_name}' to be running...")
    for _ in range(300):  # wait up to 10 minutes
        pod = core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        pod_phase = pod.status.phase
        if pod.status.init_container_statuses:
            exit_loop = False
            for init_status in pod.status.init_container_statuses:
                state = init_status.state
                if state and state.running:
                    logger.info(
                        f"Pod '{pod_name}' initContainer '{init_status.name}' is running."
                    )
                    exit_loop = True
                    break
            if exit_loop:
                break
            else:
                if pod_phase == "Running":
                    logger.info(f"Pod '{pod_name}' is running.")
                    break
                elif pod_phase in ["Failed", "Succeeded"]:
                    logger.warning(
                        f"Pod '{pod_name}' entered terminal phase: {pod_phase}"
                    )
                    break
        else:
            if pod_phase == "Running":
                logger.info(f"Pod '{pod_name}' is running.")
                break
            elif pod_phase in ["Failed", "Succeeded"]:
                logger.warning(f"Pod '{pod_name}' entered terminal phase: {pod_phase}")
                break
        time.sleep(2)
    else:
        raise RuntimeError(f"Timed out waiting for pod {pod_name} to be running")
    _check_files_in_pod(
        pod_name=pod_name,
        namespace=namespace,
        paths=[Path(data_path)],
        mount_path=data_mount_path if algorithm.meta.year <= 2024 else input_mount_path,
    )
    logger.debug(
        f"Files checked successfully in pod '{pod_name}' in namespace '{namespace}'."
    )

    if algorithm.meta.year <= 2024:
        _download_additional_files(
            algorithm=algorithm,
            pod_name=pod_name,
            namespace=namespace,
            mount_path=data_mount_path,
        )
        _upload_files_to_pod(
            pod_name=pod_name,
            namespace=namespace,
            paths=[PARAMETERS_DIR],
            mount_path=data_mount_path,
            parent_dir="parameters",
        )
    commands = ["tree", data_mount_path if algorithm.meta.year <= 2024 else input_mount_path]
    _execute_command_in_pod(
        pod_name=pod_name,
        namespace=namespace,
        command=commands,
        container="init-container",
    )
    commands = ["touch", "/etc/content_verified"]
    _execute_command_in_pod(
        pod_name=pod_name,
        namespace=namespace,
        command=commands,
        container="init-container",
    )

    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting inference")
    start_time = time.time()

    time.sleep(2)
    job_output = _observe_job_output(pod_name=pod_name, namespace=namespace)

    core_v1_api = client.CoreV1Api()
    for _ in range(300):  # Wait up to 10 minutes
        pod = core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace)
        pod_phase = pod.status.phase
        if pod_phase in ("Succeeded", "Failed"):
            logger.info(f"Finalizer pod '{pod_name}' finished with phase: {pod_phase}")
            break
        time.sleep(2)
    else:
        raise RuntimeError(
            f"Timed out waiting for finalizer pod '{pod_name}' to complete."
        )

    pvc_name_output = pvc_name
    mount_path = str(data_mount_path if algorithm.meta.year <= 2024 else input_mount_path)
    if algorithm.meta.year > 2024:
        pvc_name_output = pvc_name + "-output"
        mount_path = str(output_mount_path)

    pod_name_finalizer = _create_finalizer_job(
        job_name=job_name + "-finalizer",
        namespace=namespace,
        pvc_name=pvc_name_output,
        mount_path=mount_path,
    )
    time.sleep(2)
    _download_folder_from_pod(
        pod_name=pod_name_finalizer,
        namespace=namespace,
        container="finalizer-container",
        remote_paths=[output_mount_path],
        local_base_dir=output_path,
    )

    commands = ["touch", "/etc/content_verified"]
    _execute_command_in_pod(
        pod_name=pod_name_finalizer,
        namespace=namespace,
        command=commands,
        container="finalizer-container",
    )

    _sanity_check_output(
        data_path=data_path,
        output_path=output_path,
        container_output=job_output,
        internal_external_name_map=internal_external_name_map,
    )

    logger.debug(f"Job output: \n\r{job_output}")

    logger.info(f"Finished inference in {time.time() - start_time:.2f} seconds")
