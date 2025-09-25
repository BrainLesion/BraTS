import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
from rich.progress import Progress

from brats.core.docker import (
    _build_command_args,
    _get_container_user,
    _ensure_image,
    _get_additional_files_path,
    _get_parameters_arg,
    _get_volume_mappings_docker_only,
    _get_volume_mappings_mlcube,
    _handle_device_requests,
    _is_cuda_available,
    _log_algorithm_info,
    _observe_docker_output,
    _sanity_check_output,
    _show_docker_pull_progress,
    run_container,
)
from brats.utils.algorithm_config import AlgorithmData
from brats.constants import PARAMETERS_DIR
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
            additional_files=MagicMock(
                param_name=["weights"], param_path=["checkpoint.pth"]
            ),
            meta=MagicMock(
                challenge="Challenge",
                challenge_manuscript="challenge_manuscript_url",
                rank="1st",
                paper="paper_url",
                authors="author_names",
                dataset_manuscript="dataset_manuscript_url",
                year=2024,
            ),
        )

        self.algorithm_gpu_2025 = AlgorithmData(
            run_args=MagicMock(
                docker_image="brainles/test-image-1:latest",
                parameters_file=True,
                shm_size="1g",
                cpu_compatible=False,
            ),
            additional_files=None,
            meta=MagicMock(
                challenge="Challenge",
                challenge_manuscript="challenge_manuscript_url",
                rank="1st",
                paper="paper_url",
                authors="author_names",
                dataset_manuscript="dataset_manuscript_url",
                year=2025,
            ),
        )

        self.algorithm_cpu = AlgorithmData(
            run_args=MagicMock(
                docker_image="brainles/test-image-2:latest",
                parameters_file=True,
                shm_size="1g",
                cpu_compatible=True,
            ),
            additional_files=MagicMock(
                param_name=["weights"], param_path=["checkpoint.pth"]
            ),
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
            self.assertIn("[Downloading id1]", tasks)

            line = {
                "status": "Extracting",
                "id": "id2",
                "progressDetail": {"total": 100, "current": 50},
            }
            _show_docker_pull_progress(tasks, progress, line)
            self.assertIn("[Extracting id2]", tasks)

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
    def test_is_cuda_available_ok(self, MockRun):
        MockRun.return_value = None
        self.assertTrue(_is_cuda_available())
        MockRun.assert_called_once_with(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

    @patch("subprocess.run")
    def test_is_cuda_available_fail(self, MockRun):
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

    def test_get_volume_mappings_mlcube(self):
        result = _get_volume_mappings_mlcube(
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

    def test_get_volume_mappings_docker_only(self):
        result = _get_volume_mappings_docker_only(
            data_path=self.data_folder,
            output_path=self.output_folder,
        )
        expected = {
            str(self.data_folder.absolute()): {"bind": "/input", "mode": "rw"},
            str(self.output_folder.absolute()): {"bind": "/output", "mode": "rw"},
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

    def test_build_command_args(self):
        result = _build_command_args(self.algorithm_gpu)
        expected_command_args = [
            "--data_path=/mlcube_io0",
            "--output_path=/mlcube_io2",
            "--weights=/mlcube_io1/checkpoint.pth",
            "--parameters_file=/mlcube_io3/dummy.yml",
        ]
        for arg in expected_command_args:
            self.assertIn(arg, result)

    def test_get_container_user_requires_root(self):
        # Test when requires_root is True
        algorithm = MagicMock()
        algorithm.run_args.requires_root = True
        user = _get_container_user(algorithm)
        self.assertIsNone(user)

    @patch("brats.core.docker.os.getuid", return_value=42)
    @patch("brats.core.docker.os.getgid", return_value=1000)
    def test_get_container_user_no_root(self, MockGetGid, MockGetUid):
        # Test when requires_root is False
        algorithm = MagicMock()
        algorithm.run_args.requires_root = False
        user = _get_container_user(algorithm)
        self.assertEqual(user, "42:1000")

    @patch("brats.core.docker.Console")
    @patch("brats.core.docker.docker.models.containers.Container")
    def test_observe_docker_output(self, MockContainer, MockConsole):
        mock_container = MagicMock()
        mock_container.attach.return_value = [b"output log line"]
        mock_container.wait.return_value = {"StatusCode": 0}
        result = _observe_docker_output(mock_container)
        self.assertEqual(result, "output log line")

    @patch("brats.core.docker.logger")
    @patch("brats.core.docker.nib.load")
    def test_sanity_check_output(self, mock_nib_load, mock_logger):
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

        # Create a mock object for the fdata
        mock_nifti_img = MagicMock()
        mock_nifti_img.get_fdata.return_value = np.ones((2, 2, 2))

        # Mock the nib.load to return the mock nifti image
        mock_nib_load.return_value = mock_nifti_img

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
        mock_logger.warning.assert_not_called()

    @patch("brats.core.docker.logger")
    @patch("brats.core.docker.nib.load")
    def test_sanity_check_output_not_enough_outputs(self, mock_nib_load, mock_logger):
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

        # Create a mock object for the fdata
        mock_nifti_img = MagicMock()
        mock_nifti_img.get_fdata.return_value = np.ones((2, 2, 2))

        # Mock the nib.load to return the mock nifti image
        mock_nib_load.return_value = mock_nifti_img

        # Define container_output
        container_output = "Sample container output"

        # Check that the exception is raised
        with self.assertRaises(BraTSContainerException):
            _sanity_check_output(
                data_path=mock_data_path,
                output_path=mock_output_path,
                container_output=container_output,
            )
            mock_logger.assert_not_called()

    @patch("brats.core.docker.logger")
    @patch("brats.core.docker.nib.load")
    def test_sanity_check_output_empty_warning_single_inference(
        self, mock_nib_load, mock_logger
    ):
        # Create mock paths
        mock_data_path = MagicMock(spec=Path)
        mock_output_path = MagicMock(spec=Path)

        # Simulate input files starting with "BraTS" and output files
        mock_data_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
        ]
        mock_output_path.iterdir.return_value = [
            MagicMock(name="file1", spec=Path),
        ]

        # Create a mock object for the fdata
        mock_nifti_img = MagicMock()
        # zeros!
        mock_nifti_img.get_fdata.return_value = np.zeros((2, 2, 2))

        # Mock the nib.load to return the mock nifti image
        mock_nib_load.return_value = mock_nifti_img

        # Define container_output
        container_output = "Sample container output"

        # Check that no exception is raised

        _sanity_check_output(
            data_path=mock_data_path,
            output_path=mock_output_path,
            container_output=container_output,
        )

        # assertions
        mock_logger.warning.assert_called_once()

    @patch("brats.core.docker.logger")
    @patch("brats.core.docker.nib.load")
    def test_sanity_check_output_empty_warning_batch_inference(
        self, mock_nib_load, mock_logger
    ):
        # Create mock paths
        mock_data_path = MagicMock(spec=Path)
        mock_output_path = MagicMock(spec=Path)

        # Simulate input files starting with "BraTS" and output files
        mock_data_path.iterdir.return_value = [
            MagicMock(name="external_file_1", spec=Path),
        ]
        mock_output_path.iterdir.return_value = [
            MagicMock(name="internal_file_1", spec=Path),
        ]

        # Create a mock object for the fdata
        mock_nifti_img = MagicMock()
        # zeros!
        mock_nifti_img.get_fdata.return_value = np.zeros((2, 2, 2))

        # Mock the nib.load to return the mock nifti image
        mock_nib_load.return_value = mock_nifti_img

        # Define container_output
        container_output = "Sample container output"

        # Check that no exception is raised

        _sanity_check_output(
            data_path=mock_data_path,
            output_path=mock_output_path,
            container_output=container_output,
            internal_external_name_map={"internal_file_1": "external_file_1"},
        )

        # assertions
        mock_logger.warning.assert_called_once()

    @patch("brats.core.docker.logger.debug")
    def test_log_algorithm_info(self, MockLoggerDebug):
        _log_algorithm_info(algorithm=self.algorithm_gpu)

        MockLoggerDebug.assert_called_once()

    @patch("brats.core.docker._log_algorithm_info")
    @patch("brats.core.docker._ensure_image")
    @patch("brats.core.docker._get_additional_files_path")
    @patch("brats.core.docker._get_container_user")
    @patch("brats.core.docker._get_volume_mappings_mlcube")
    @patch("brats.core.docker._build_command_args")
    @patch("brats.core.docker._handle_device_requests")
    @patch("brats.core.docker._observe_docker_output")
    @patch("brats.core.docker.client")
    def test_run_container_mlcube(
        self,
        mock_client,
        mock_observe_docker_output,
        mock_handle_device_requests,
        mock_build_command_args,
        mock_get_volume_mappings_mlcube,
        mock_get_container_user,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
    ):

        # setup mocks
        mock_build_command_args.return_value = "args"

        # run
        cuda_devices = "0"
        force_cpu = False
        run_container(
            algorithm=self.algorithm_gpu,
            data_path=self.data_folder,
            output_path=self.output_folder,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )

        # Verify mocks were called as expected
        mock_log_algorithm_info.assert_called_once_with(algorithm=self.algorithm_gpu)
        mock_ensure_image.assert_called_once()
        mock_get_additional_files_path.assert_called_once()
        mock_get_container_user.assert_called_once()
        mock_get_volume_mappings_mlcube.assert_called_once()
        mock_build_command_args.assert_called_once()
        mock_handle_device_requests.assert_called_once()

    @patch("brats.core.docker._log_algorithm_info")
    @patch("brats.core.docker._ensure_image")
    @patch("brats.core.docker._get_additional_files_path")
    @patch("brats.core.docker._get_container_user")
    @patch("brats.core.docker._get_volume_mappings_docker_only")
    @patch("brats.core.docker._handle_device_requests")
    @patch("brats.core.docker._observe_docker_output")
    @patch("brats.core.docker.client")
    def test_run_container_docker_only(
        self,
        mock_client,
        mock_observe_docker_output,
        mock_handle_device_requests,
        mock_get_volume_mappings_docker_only,
        mock_get_container_user,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
    ):
        # run
        cuda_devices = "0"
        force_cpu = False
        run_container(
            algorithm=self.algorithm_gpu_2025,
            data_path=self.data_folder,
            output_path=self.output_folder,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )

        # Verify mocks were called as expected
        mock_log_algorithm_info.assert_called_once_with(
            algorithm=self.algorithm_gpu_2025
        )
        mock_ensure_image.assert_called_once()
        mock_get_additional_files_path.assert_called_once()
        mock_get_container_user.assert_called_once()
        mock_get_volume_mappings_docker_only.assert_called_once()
        mock_handle_device_requests.assert_called_once()
