import io
import tarfile
import base64
import types
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Target module
import brats.core.kubernetes as k8s


@pytest.fixture
def dummy_algorithm():
    class AddFiles:
        def __init__(self):
            self.record_id = "12345"
            self.param_name = ["weights", "config"]
            self.param_path = ["w.bin", "c.yaml"]

    class Meta:
        def __init__(self, year):
            self.year = year

    class RunArgs:
        def __init__(self):
            self.docker_image = "alpine:latest"
            self.shm_size = "1gb"
            self.parameters_file = None

    class Algo:
        def __init__(self, year=2024, with_additional=True):
            self.meta = Meta(year)
            self.run_args = RunArgs()
            self.additional_files = AddFiles() if with_additional else None

    return Algo


@pytest.fixture
def tmp_tree(tmp_path):
    # creates input dir structure with one file
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "case1").mkdir(parents=True, exist_ok=True)
    f = input_dir / "case1" / "image.nii.gz"
    f.write_bytes(b"dummy")
    return tmp_path


def _mk_pod(name="pod-1", phase="Running", init_running=True, creation_ts=None):
    pod = types.SimpleNamespace()
    pod.metadata = types.SimpleNamespace()
    pod.metadata.name = name
    pod.metadata.creation_timestamp = creation_ts or datetime.now(timezone.utc)
    pod.status = types.SimpleNamespace()
    pod.status.phase = phase
    # container_statuses for _observe_job_output
    cstat = types.SimpleNamespace()
    cstate = types.SimpleNamespace()
    cstate.running = True
    cstat.name = "job-container"
    cstat.state = cstate
    pod.status.container_statuses = [cstat]
    # init container statuses for run_job wait loop
    istat = types.SimpleNamespace()
    istat.name = "init-container"
    istate = types.SimpleNamespace()
    istate.running = True if init_running else None
    istat.state = istate
    pod.status.init_container_statuses = [istat] if init_running is not None else None
    return pod


### _build_command_args


def test_build_command_args_with_additional_and_params(
    dummy_algorithm, monkeypatch, tmp_path
):
    algo = dummy_algorithm(year=2024, with_additional=True)
    monkeypatch.setattr(
        k8s, "_get_parameters_arg", lambda algorithm: " --params=/mlcube_io3/foo.json"
    )

    cmd = k8s._build_command_args(
        algorithm=algo,
        additional_files_path="/data/weights_dir",
        data_path="/data/input",
        output_path="/data/output",
        mount_path="/data",
    )

    # must include basic paths
    assert "--data_path=/data/input" in cmd
    assert "--output_path=/data/output" in cmd
    # must include additional files args with per-param path
    assert "--weights=/data/weights_dir/w.bin" in cmd
    assert "--config=/data/weights_dir/c.yaml" in cmd
    # must rewrite /mlcube_io3 to /data/parameters
    assert "/mlcube_io3" not in cmd
    assert "/data/parameters/foo.json" in cmd


def test_build_command_args_without_additional_and_params(dummy_algorithm, monkeypatch):
    algo = dummy_algorithm(year=2024, with_additional=False)
    monkeypatch.setattr(k8s, "_get_parameters_arg", lambda algorithm: "")

    cmd = k8s._build_command_args(
        algorithm=algo,
        additional_files_path="/ignored",
        data_path="/data/input",
        output_path="/data/output",
        mount_path="/data",
    )

    assert "--data_path=/data/input" in cmd
    assert "--output_path=/data/output" in cmd
    assert "--weights=" not in cmd
    assert "--config=" not in cmd


### _execute_command_in_pod


def test_execute_command_in_pod(monkeypatch):
    mock_core = MagicMock()
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)
    mock_stream = MagicMock(return_value="OK")
    monkeypatch.setattr(k8s, "stream", mock_stream)

    out = k8s._execute_command_in_pod(
        pod_name="p", namespace="ns", command=["echo", "hi"], container="job-container"
    )

    assert out == "OK"
    mock_stream.assert_called_once()
    # ensure connect_get_namespaced_pod_exec passed into stream
    assert mock_stream.call_args.kwargs.get("_preload_content") is True


### _download_additional_files


def test_download_additional_files_success(dummy_algorithm, monkeypatch):
    algo = dummy_algorithm(year=2024, with_additional=True)
    monkeypatch.setattr(
        k8s,
        "_get_zenodo_metadata_and_archive_url",
        lambda record_id: ({"version": "2"}, "http://example/archive.zip"),
    )
    exec_calls = []

    def fake_exec(**kwargs):
        exec_calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(k8s, "_execute_command_in_pod", lambda **kw: fake_exec(**kw))
    p = k8s._download_additional_files(
        algorithm=algo, pod_name="p", namespace="ns", mount_path="/data"
    )
    assert str(p) == "/data/12345_v2"
    assert exec_calls and exec_calls[0]["container"] == "init-container"


def test_download_additional_files_zenodo_failure(dummy_algorithm, monkeypatch):
    algo = dummy_algorithm(year=2024, with_additional=True)
    monkeypatch.setattr(
        k8s, "_get_zenodo_metadata_and_archive_url", lambda record_id: None
    )
    with pytest.raises(k8s.ZenodoException):
        k8s._download_additional_files(
            algorithm=algo, pod_name="p", namespace="ns", mount_path="/data"
        )


### _check_files_in_pod


def test_check_files_in_pod_uploads_missing(monkeypatch, tmp_tree):
    # Simulate ls output saying missing
    monkeypatch.setattr(
        k8s, "_execute_command_in_pod", lambda **kw: "No such file or directory"
    )
    uploaded = {"called": False, "args": None}

    def fake_upload(**kwargs):
        uploaded["called"] = True
        uploaded["args"] = kwargs

    monkeypatch.setattr(k8s, "_upload_files_to_pod", lambda **kw: fake_upload(**kw))

    k8s._check_files_in_pod(
        pod_name="p", namespace="ns", paths=[tmp_tree / "input"], mount_path="/data"
    )

    assert uploaded["called"] is True
    assert uploaded["args"]["parent_dir"] == "input"


### _download_folder_from_pod


def test_download_folder_from_pod(monkeypatch, tmp_path):
    # Create a tar stream that contains one file "foo.txt"
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        data = io.BytesIO(b"hello")
        info = tarfile.TarInfo(name="foo.txt")
        info.size = len(b"hello")
        tar.addfile(info, data)
    tar_bytes.seek(0)
    b64 = base64.b64encode(tar_bytes.getvalue())

    # Make fake streaming response
    class FakeResp:
        def __init__(self):
            self._stdout_chunks = [b64]
            self._stderr_chunks = []
            self._open = True

        def is_open(self):
            return self._open

        def update(self, timeout=1):
            # close after delivering stdout
            self._open = False

        def peek_stdout(self):
            return bool(self._stdout_chunks)

        def read_stdout(self):
            return self._stdout_chunks.pop(0)

        def peek_stderr(self):
            return False

        def read_stderr(self):
            return ""

        def close(self):
            self._open = False

    monkeypatch.setattr(k8s, "_execute_command_in_pod", lambda **kw: FakeResp())

    k8s._download_folder_from_pod(
        pod_name="p",
        namespace="ns",
        container="finalizer-container",
        remote_paths=[Path("/output")],
        local_base_dir=tmp_path,
    )

    # Expect foo.txt extracted into tmp_path
    expect = tmp_path / "foo.txt"
    assert expect.exists()
    assert expect.read_text() == "hello"


### _upload_files_to_pod


def test_upload_files_to_pod(monkeypatch, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")

    class FakeResp:
        def __init__(self):
            self.stdin = io.BytesIO()

        def write_stdin(self, data):
            # consume some bytes
            assert isinstance(data, (bytes, bytearray))

        def close(self):
            pass

    monkeypatch.setattr(k8s, "_execute_command_in_pod", lambda **kw: FakeResp())

    k8s._upload_files_to_pod(
        pod_name="p",
        namespace="ns",
        paths=[f],
        mount_path="/data",
        relative_to=tmp_path,
        parent_dir="input",
    )


### _create_namespaced_pvc


def test_create_pvc_skips_if_exists(monkeypatch):
    mock_core = MagicMock()
    pvc = types.SimpleNamespace()
    pvc.metadata = types.SimpleNamespace(name="my-pvc")
    mock_core.list_namespaced_persistent_volume_claim.return_value = (
        types.SimpleNamespace(items=[pvc])
    )
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    k8s._create_namespaced_pvc("my-pvc", "default", "5Gi", None)
    mock_core.create_namespaced_persistent_volume_claim.assert_not_called()


def test_create_pvc_creates_with_and_without_storage_class(monkeypatch):
    mock_core = MagicMock()
    mock_core.list_namespaced_persistent_volume_claim.return_value = (
        types.SimpleNamespace(items=[])
    )
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    k8s._create_namespaced_pvc("p1", "ns", "1Gi", None)
    k8s._create_namespaced_pvc("p2", "ns", "2Gi", "fast")

    assert mock_core.create_namespaced_persistent_volume_claim.call_count == 2


### _create_finalizer_job


def test_create_finalizer_job(monkeypatch):
    mock_batch = MagicMock()
    mock_core = MagicMock()

    # No existing jobs
    mock_batch.list_namespaced_job.return_value = types.SimpleNamespace(items=[])
    # pods appear after first poll
    pod = _mk_pod(name="final-pod", phase="Running", init_running=None)
    mock_core.list_namespaced_pod.side_effect = [
        types.SimpleNamespace(items=[]),
        types.SimpleNamespace(items=[pod]),
    ]

    monkeypatch.setattr(k8s.client, "BatchV1Api", lambda: mock_batch)
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    name = k8s._create_finalizer_job(
        job_name="job-final", namespace="ns", pvc_name="pvc", mount_path="/data"
    )
    assert name == "final-pod"
    assert mock_batch.create_namespaced_job.call_count == 1


### _create_namespaced_job


def test_create_namespaced_job_deletes_old_pod_and_creates_new(monkeypatch):
    mock_batch = MagicMock()
    mock_core = MagicMock()

    # Existing job with same name to be deleted
    mock_batch.list_namespaced_job.return_value = types.SimpleNamespace(
        items=[types.SimpleNamespace(metadata=types.SimpleNamespace(name="job-1"))]
    )

    # Existing pod to be deleted
    old_pod = _mk_pod(name="old-pod")
    mock_core.list_namespaced_pod.return_value = types.SimpleNamespace(items=[old_pod])

    # When polling for new pod, first empty then one appears
    new_pod = _mk_pod(name="new-pod")
    mock_core.list_namespaced_pod.side_effect = [
        types.SimpleNamespace(items=[old_pod]),  # for deletion scan
        types.SimpleNamespace(items=[]),  # first poll
        types.SimpleNamespace(items=[new_pod]),  # second poll -> found
    ]

    monkeypatch.setattr(k8s.client, "BatchV1Api", lambda: mock_batch)
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    name = k8s._create_namespaced_job(
        job_name="job-1",
        namespace="ns",
        pvc_name="pvc",
        image="alpine:latest",
        device_requests=[],
        pv_mounts={"pvc": "/data"},
        args=["echo", "hi"],
        shm_size="1gb",
        user=None,
    )

    assert name == "new-pod"
    assert mock_batch.delete_namespaced_job.call_count == 1
    assert mock_core.delete_namespaced_pod.call_count == 1
    assert mock_batch.create_namespaced_job.call_count == 1


### run_job (two branches)


def test_run_job_year_2024_flow(monkeypatch, tmp_tree, tmp_path, dummy_algorithm):
    algo = dummy_algorithm(year=2024, with_additional=True)

    # Config load
    monkeypatch.setattr(k8s.config, "load_kube_config", lambda: None)

    # Helper functions
    monkeypatch.setattr(k8s, "_log_algorithm_info", lambda **kw: None)
    monkeypatch.setattr(k8s, "_handle_device_requests", lambda **kw: [])
    monkeypatch.setattr(k8s, "_get_container_user", lambda **kw: None)
    monkeypatch.setattr(k8s, "_get_parameters_arg", lambda algorithm=None: "")
    monkeypatch.setattr(
        k8s,
        "_get_zenodo_metadata_and_archive_url",
        lambda record_id: ({"version": "1"}, "http://example/zip"),
    )
    monkeypatch.setattr(k8s, "PARAMETERS_DIR", tmp_tree / "params")
    (tmp_tree / "params").mkdir(exist_ok=True)
    monkeypatch.setattr(k8s, "get_dummy_path", lambda: Path("/dummy"))
    monkeypatch.setattr(k8s, "_sanity_check_output", lambda **kw: None)

    # PVC creation
    monkeypatch.setattr(k8s, "_create_namespaced_pvc", lambda **kw: None)

    # Job + pod lifecycle
    def fake_create_job(**kwargs):
        return "job-pod"

    monkeypatch.setattr(
        k8s, "_create_namespaced_job", lambda **kw: fake_create_job(**kw)
    )

    # CoreV1Api read_namespaced_pod progression
    mock_core = MagicMock()
    # First loop waits until init container running
    mock_core.read_namespaced_pod.return_value = _mk_pod(
        name="job-pod", phase="Running", init_running=True
    )
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    # File checking
    monkeypatch.setattr(k8s, "_check_files_in_pod", lambda **kw: None)

    # Additional files download and parameters upload
    monkeypatch.setattr(
        k8s, "_download_additional_files", lambda **kw: Path("/data/12345_v1")
    )
    monkeypatch.setattr(k8s, "_upload_files_to_pod", lambda **kw: None)

    # Exec commands in init container
    monkeypatch.setattr(k8s, "_execute_command_in_pod", lambda **kw: "ok")

    # Observe logs
    monkeypatch.setattr(k8s, "_observe_job_output", lambda **kw: "LOGS")

    # Pod completes
    def read_pod_done(name, namespace):
        return _mk_pod(name=name, phase="Succeeded", init_running=True)

    mock_core.read_namespaced_pod.side_effect = [
        _mk_pod(name="job-pod", phase="Running", init_running=True),  # wait loop
        _mk_pod(
            name="job-pod", phase="Succeeded", init_running=True
        ),  # completion loop
    ]

    # Finalizer job
    monkeypatch.setattr(k8s, "_create_finalizer_job", lambda **kw: "final-pod")
    monkeypatch.setattr(k8s, "_download_folder_from_pod", lambda **kw: None)

    out_dir = tmp_path / "out"
    k8s.run_job(
        algorithm=algo,
        data_path=tmp_tree / "input",
        output_path=out_dir,
        cuda_devices="",
        force_cpu=True,
        internal_external_name_map=None,
        namespace="default",
        pvc_name="mypvc",
        pvc_storage_size="1Gi",
        pvc_storage_class=None,
        job_name="myjob",
        data_mount_path="/data",
    )

    # Ensure output dir created
    assert out_dir.exists()


def test_run_job_year_2025_flow(monkeypatch, tmp_tree, tmp_path, dummy_algorithm):
    algo = dummy_algorithm(year=2025, with_additional=False)
    monkeypatch.setattr(k8s.config, "load_kube_config", lambda: None)
    monkeypatch.setattr(k8s, "_log_algorithm_info", lambda **kw: None)
    monkeypatch.setattr(k8s, "_handle_device_requests", lambda **kw: [])
    monkeypatch.setattr(k8s, "_get_container_user", lambda **kw: None)
    monkeypatch.setattr(k8s, "_get_parameters_arg", lambda algorithm=None: "")
    monkeypatch.setattr(k8s, "_sanity_check_output", lambda **kw: None)
    # PVCs
    created = []

    def fake_create_pvc(**kw):
        created.append(kw["pvc_name"])

    monkeypatch.setattr(
        k8s, "_create_namespaced_pvc", lambda **kw: fake_create_pvc(**kw)
    )

    # Job creation
    monkeypatch.setattr(k8s, "_create_namespaced_job", lambda **kw: "job-pod")

    # Pod running/completion
    mock_core = MagicMock()
    mock_core.read_namespaced_pod.side_effect = [
        _mk_pod(name="job-pod", phase="Running", init_running=True),
        _mk_pod(name="job-pod", phase="Succeeded", init_running=True),
    ]
    monkeypatch.setattr(k8s.client, "CoreV1Api", lambda: mock_core)

    # File checks skipped in behavior but called; mock anyway
    monkeypatch.setattr(k8s, "_check_files_in_pod", lambda **kw: None)
    monkeypatch.setattr(k8s, "_execute_command_in_pod", lambda **kw: "ok")
    monkeypatch.setattr(k8s, "_observe_job_output", lambda **kw: "LOGS")
    monkeypatch.setattr(k8s, "_create_finalizer_job", lambda **kw: "final-pod")
    monkeypatch.setattr(k8s, "_download_folder_from_pod", lambda **kw: None)

    out_dir = tmp_path / "out2"
    k8s.run_job(
        algorithm=algo,
        data_path=tmp_tree / "input",
        output_path=out_dir,
        cuda_devices="",
        force_cpu=True,
        namespace="default",
        pvc_name="mypvc2",
        pvc_storage_size="1Gi",
        job_name="myjob2",
        data_mount_path="/data",
    )

    # For year > 2024, an extra output PVC should be created
    assert "mypvc2" in created
    assert "mypvc2-output" in created
    assert out_dir.exists()
