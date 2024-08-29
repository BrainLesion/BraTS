import unittest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import shutil
import subprocess
import tempfile

from brats.core.docker import (
    _show_docker_pull_progress,
    _ensure_image,
    _is_cuda_available,
    _handle_device_requests,
    _get_additional_files_path,
    _get_volume_mappings,
    _get_parameters_arg,
    _build_args,
    _observe_docker_output,
    _sanity_check_output,
)
from rich.progress import Progress
from brats.utils.algorithm_config import AlgorithmData
from brats.utils.constants import PARAMETERS_DIR
from brats.utils.exceptions import (
    AlgorithmNotCPUCompatibleException,
    BraTSContainerException,
)


class TestDockerHelpers(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_folder = self.test_dir / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder = self.test_dir / "output"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Create mock algorithm data
        self.algorithm_gpu = AlgorithmData(
            run_args=MagicMock(
                docker_image="brainles/test-image-1:latest",
                parameters_file=True,
                shm_size="1g",
                cpu_compatible=False,
            ),
            weights=MagicMock(param_name="weights", checkpoint_path="checkpoint.pth"),
            meta=MagicMock(
                challenge="Challenge",
                rank="1st",
                paper="paper_url",
                authors="author_names",
            ),
        )

        self.algorithm_cpu = AlgorithmData(
            run_args=MagicMock(
                docker_image="brainles/test-image-2:latest",
                parameters_file=True,
                shm_size="1g",
                cpu_compatible=True,
            ),
            weights=MagicMock(param_name="weights", checkpoint_path="checkpoint.pth"),
            meta=MagicMock(
                challenge="Challenge",
                rank="1st",
                paper="paper_url",
                authors="author_names",
            ),
        )

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch("brats.core.docker.client")
    def test_show_docker_pull_progress(self, MockClient):
        tasks = {}
        with Progress() as progress:
            line = {
                "status": "Downloading",
                "id": "id1",
                "progressDetail": {"total": 100, "current": 50},
            }
            _show_docker_pull_progress(tasks, progress, line)
            self.assertIn("[Download id1]", tasks)

            line = {
                "status": "Extracting",
                "id": "id2",
                "progressDetail": {"total": 100, "current": 50},
            }
            _show_docker_pull_progress(tasks, progress, line)
            self.assertIn("[Extract id2]", tasks)

    @patch("brats.core.docker.client.images.list", return_value=[])
    @patch("brats.core.docker.client.api.pull")
    def test_ensure_image(self, MockPull, MockList):
        MockPull.return_value = iter(
            [
                {
                    "status": "Downloading",
                    "id": "test_image",
                    "progressDetail": {"total": 100, "current": 50},
                }
            ]
        )
        _ensure_image("test-image:latest")
        MockPull.assert_called_once_with("test-image:latest", stream=True, decode=True)

    @patch("subprocess.run")
    def test_is_cuda_available(self, MockRun):
        MockRun.return_value = None
        self.assertTrue(_is_cuda_available())
        MockRun.assert_called_once_with(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    @patch("subprocess.run")
    def test_is_cuda_available(self, MockRun):
        MockRun.side_effect = Exception()
        self.assertFalse(_is_cuda_available())
        MockRun.assert_called_once_with(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    @patch("brats.core.docker._is_cuda_available", return_value=True)
    def test_handle_device_requests_cuda(self, MockIsCudaAvailable):
        result = _handle_device_requests(
            algorithm=self.algorithm_gpu, cuda_devices="42", force_cpu=False
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].device_ids, ["42"])
        self.assertEqual(result[0].capabilities, [["gpu"]])

    @patch("brats.core.docker._is_cuda_available", return_value=False)
    def test_handle_device_requests_force_cpu_valid(self, MockIsCudaAvailable):
        device_requests = _handle_device_requests(
            algorithm=self.algorithm_cpu, cuda_devices="0", force_cpu=True
        )
        self.assertEqual(len(device_requests), 0)

    @patch("brats.core.docker._is_cuda_available", return_value=False)
    def test_handle_device_requests_no_cuda_no_cpu(self, MockIsCudaAvailable):
        with self.assertRaises(AlgorithmNotCPUCompatibleException):
            _handle_device_requests(
                algorithm=self.algorithm_gpu, cuda_devices="0", force_cpu=False
            )

    @patch("brats.core.docker._is_cuda_available", return_value=False)
    def test_handle_device_requests_force_cpu_invalid(self, MockIsCudaAvailable):
        with self.assertRaises(AlgorithmNotCPUCompatibleException):
            _handle_device_requests(
                algorithm=self.algorithm_gpu, cuda_devices="0", force_cpu=True
            )

    @patch("brats.core.docker.check_additional_files_path")
    def test_get_additional_files_path(self, MockCheckAdditionalFilesPath):
        MockCheckAdditionalFilesPath.return_value = self.test_dir
        result = _get_additional_files_path(self.algorithm_gpu)
        self.assertEqual(result, self.test_dir)

    def test_get_volume_mappings(self):
        result = _get_volume_mappings(
            data_path=self.data_folder,
            additional_files_path=self.test_dir,
            output_path=self.output_folder,
            parameters_path=PARAMETERS_DIR,
        )
        expected = {
            self.data_folder.absolute(): {"bind": "/mlcube_io0", "mode": "rw"},
            self.test_dir.absolute(): {"bind": "/mlcube_io1", "mode": "rw"},
            self.output_folder.absolute(): {"bind": "/mlcube_io2", "mode": "rw"},
            PARAMETERS_DIR.absolute(): {"bind": "/mlcube_io3", "mode": "rw"},
        }
        self.assertEqual(result, expected)

    def test_get_parameters_arg_dummy(self):
        result = _get_parameters_arg(self.algorithm_gpu)
        expected = f" --parameters_file=/mlcube_io3/dummy.yml"
        self.assertEqual(result, expected)

    def test_get_parameters_arg_file(self):
        with patch("brats.core.docker.PARAMETERS_DIR", self.test_dir):
            identifier = self.algorithm_gpu.run_args.docker_image.split(":")[0].split(
                "/"
            )[-1]
            file = self.test_dir / f"{identifier}.yml"
            file.touch()
            result = _get_parameters_arg(self.algorithm_gpu)
            expected = f" --parameters_file=/mlcube_io3/{file.name}"
            self.assertEqual(result, expected)

    def test_build_args(self):
        result = _build_args(self.algorithm_gpu)
        expected_command_args = [
            "--data_path=/mlcube_io0",
            "--output_path=/mlcube_io2",
            "--weights=/mlcube_io1/checkpoint.pth",
            "--parameters_file=/mlcube_io3/dummy.yml",
        ]
        for arg in expected_command_args:
            self.assertIn(arg, result[0])
        self.assertEqual(result[1], {})

    @patch("brats.core.docker.Console")
    @patch("brats.core.docker.docker.models.containers.Container")
    def test_observe_docker_output(self, MockContainer, MockConsole):
        mock_container = MagicMock()
        mock_container.attach.return_value = [b"output log line"]
        mock_container.wait.return_value = {"StatusCode": 0}
        result = _observe_docker_output(mock_container)
        self.assertEqual(result, "output log line")

    def test_sanity_check_output(self):
        # Create mock paths
        mock_data_path = MagicMock(spec=Path)
        mock_output_path = MagicMock(spec=Path)

        # Simulate input files starting with "BraTS" and output files
        mock_data_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
            MagicMock(name="file2", spec=Path),
        ]
        mock_output_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
            MagicMock(name="file2", spec=Path),
        ]

        # Define container_output
        container_output = "Sample container output"

        # Check that no exception is raised
        try:
            _sanity_check_output(
                data_path=mock_data_path,
                output_path=mock_output_path,
                container_output=container_output,
            )
        except BraTSContainerException:
            self.fail("BraTSContainerException was raised unexpectedly")

    def test_sanity_check_output_fail(self):
        # Create mock paths
        mock_data_path = MagicMock(spec=Path)
        mock_output_path = MagicMock(spec=Path)

        # Simulate input files starting with "BraTS" and output files
        mock_data_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
            MagicMock(name="file2", spec=Path),
        ]
        mock_output_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
        ]

        # Define container_output
        container_output = "Sample container output"

        # Check that the exception is raised
        with self.assertRaises(BraTSContainerException):
            _sanity_check_output(
                data_path=mock_data_path,
                output_path=mock_output_path,
                container_output=container_output,
            )
