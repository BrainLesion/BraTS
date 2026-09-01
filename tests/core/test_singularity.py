import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from brats.core.singularity import (
    _build_command_args,
    _convert_volume_mappings_to_singularity_format,
    _ensure_image,
    _get_docker_working_dir,
    run_container,
)
from brats.utils.algorithm_config import AlgorithmData


class TestSingularityHelpers(unittest.TestCase):
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
                year=2023,
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
                year=2025,
            ),
        )

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity.logger")
    @patch("brats.core.singularity.Path.exists")
    @patch("tempfile.gettempdir")
    def test_ensure_image_pulls_if_missing(
        self, MockGetTempDir, MockExists, MockLogger, MockPull
    ):
        # Arrange: simulate missing image file
        MockExists.return_value = False
        MockGetTempDir.return_value = self.test_dir
        fake_image_path = str(
            self.test_dir / "brats_singularity_images" / "test-image_latest"
        )
        fake_image = "test-image:latest"

        result = _ensure_image(fake_image)
        # Assert
        MockPull.assert_called_once_with(
            [
                "singularity",
                "build",
                "--sandbox",
                "--fakeroot",
                fake_image_path,
                "docker://" + fake_image,
            ],
            check=True,
        )
        assert result == fake_image_path
        MockLogger.debug.assert_any_call(
            f"Pulling Singularity image {fake_image} and creating a Sandbox at {fake_image_path}"
        )

    @patch("brats.core.singularity._ensure_docker_image")
    @patch("brats.core.singularity.docker_client")
    def test_get_working_dir_from_docker_image(self, MockDockerClient, MockEnsureImage):
        image = "brainles/test-image_latest"
        MockEnsureImage.return_value = image
        MockDockerClient.images.get.return_value = MagicMock(
            attrs={"Config": {"WorkingDir": "/workspace"}}
        )

        working_dir = _get_docker_working_dir(image)
        self.assertEqual(working_dir, Path("/workspace"))

    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity.Path.exists")
    @patch("tempfile.gettempdir")
    def test_ensure_image_returns_if_exists(self, MockGetTempDir, MockExists, MockPull):
        # Arrange: simulate existing image file
        MockExists.return_value = True
        MockGetTempDir.return_value = self.test_dir
        fake_image_path = str(
            self.test_dir / "brats_singularity_images" / "fake_image_latest"
        )

        result = _ensure_image("fake_image:latest")

        # Assert
        assert result == fake_image_path
        MockPull.assert_not_called()
        # puller should not be consumed since image exists

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

    def test_build_command_args_param_name_none(self):
        algorithm = AlgorithmData(
            run_args=MagicMock(
                docker_image="brainles/test-image:latest",
                parameters_file=True,
                shm_size="1g",
                cpu_compatible=False,
            ),
            additional_files=MagicMock(
                param_name=None,
                param_path=[],
            ),
            meta=MagicMock(
                challenge="Challenge",
                rank="1st",
                paper="paper_url",
                authors="author_names",
            ),
        )
        result = _build_command_args(algorithm)
        self.assertIn("--data_path=/mlcube_io0", result)
        self.assertIn("--output_path=/mlcube_io2", result)
        self.assertIn("--parameters_file=/mlcube_io3/dummy.yml", result)
        self.assertNotIn("--weights", result)

    def test_convert_volume_mappings_to_singularity_format(self):
        result = _convert_volume_mappings_to_singularity_format(
            volume_mappings={
                str(self.data_folder.absolute()): {"bind": "/input", "mode": "rw"},
                str(self.output_folder.absolute()): {"bind": "/output", "mode": "rw"},
            }
        )
        expected = [
            f"{self.data_folder.absolute()}:/input",
            f"{self.output_folder.absolute()}:/output",
        ]
        self.assertEqual(result, expected)

    @patch("brats.core.singularity._log_algorithm_info")
    @patch("brats.core.singularity._ensure_image")
    @patch("brats.core.singularity._get_additional_files_path")
    @patch("brats.core.singularity._get_volume_mappings_mlcube")
    @patch("brats.core.singularity._build_command_args")
    @patch("brats.core.singularity._handle_device_requests")
    @patch("brats.core.singularity._convert_volume_mappings_to_singularity_format")
    @patch("brats.core.singularity.Client")
    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity._get_docker_working_dir")
    def test_run_singularity_container(
        self,
        mock_get_docker_working_dir,
        mock_subprocess_run,
        mock_client,
        mock_convert_volume_mappings_to_singularity_format,
        mock_handle_device_requests,
        mock_build_command_args,
        mock_get_volume_mappings_mlcube,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
    ):
        mock_handle_device_requests.return_value = []
        # setup mocks
        mock_build_command_args.return_value = [
            "--data_path=/mlcube_io0",
            "--output_path=/mlcube_io2",
            "--weights=/mlcube_io1/checkpoint.pth",
            "--parameters_file=/mlcube_io3/dummy.yml",
        ]

        # run
        cuda_devices = "0"
        force_cpu = False
        mock_client.run.return_value = iter([])

        mock_ensure_image.return_value = str(
            self.test_dir / "brats_singularity_images" / "brainles_test-image_latest"
        )
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
        mock_build_command_args.assert_called_once()
        mock_get_volume_mappings_mlcube.assert_called_once()
        mock_handle_device_requests.assert_called_once()
        mock_convert_volume_mappings_to_singularity_format.assert_called_once()
        mock_get_docker_working_dir.assert_called_once()
        mock_subprocess_run.assert_called_once()
        mock_client.run.assert_called_once()

    @patch("brats.core.singularity.logger")
    @patch("brats.core.singularity._log_algorithm_info")
    @patch("brats.core.singularity._ensure_image")
    @patch("brats.core.singularity._get_additional_files_path")
    @patch("brats.core.singularity._get_volume_mappings_mlcube")
    @patch("brats.core.singularity._build_command_args")
    @patch("brats.core.singularity._handle_device_requests")
    @patch("brats.core.singularity._convert_volume_mappings_to_singularity_format")
    @patch("brats.core.singularity.Client")
    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity._get_docker_working_dir")
    def test_run_container_sets_cuda_visible_devices(
        self,
        mock_get_docker_working_dir,
        mock_subprocess_run,
        mock_client,
        mock_convert_volume_mappings_to_singularity_format,
        mock_handle_device_requests,
        mock_build_command_args,
        mock_get_volume_mappings_mlcube,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
        mock_logger,
    ):
        mock_handle_device_requests.return_value = [MagicMock()]  # GPU requested
        mock_build_command_args.return_value = ["--data_path=/mlcube_io0"]
        mock_ensure_image.return_value = str(
            self.test_dir / "brats_singularity_images" / "brainles_test-image_latest"
        )

        captured_env = {}

        def capture_env(*args, **kwargs):
            captured_env["value"] = os.environ.get(
                "SINGULARITYENV_CUDA_VISIBLE_DEVICES"
            )
            return iter([])

        mock_client.run.side_effect = capture_env

        run_container(
            algorithm=self.algorithm_gpu,
            data_path=self.data_folder,
            output_path=self.output_folder,
            cuda_devices="0,1",
            force_cpu=False,
        )

        # the container sees only the requested devices while running
        self.assertEqual(captured_env["value"], "0,1")
        options = mock_client.run.call_args.kwargs["options"]
        self.assertIn("--nv", options)
        # environment is restored after the run
        self.assertNotIn("SINGULARITYENV_CUDA_VISIBLE_DEVICES", os.environ)

    @patch("brats.core.singularity.logger")
    @patch("brats.core.singularity._log_algorithm_info")
    @patch("brats.core.singularity._ensure_image")
    @patch("brats.core.singularity._get_additional_files_path")
    @patch("brats.core.singularity._get_volume_mappings_mlcube")
    @patch("brats.core.singularity._build_command_args")
    @patch("brats.core.singularity._handle_device_requests")
    @patch("brats.core.singularity._convert_volume_mappings_to_singularity_format")
    @patch("brats.core.singularity.Client")
    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity._get_docker_working_dir")
    def test_run_container_overrides_and_restores_existing_cuda_env(
        self,
        mock_get_docker_working_dir,
        mock_subprocess_run,
        mock_client,
        mock_convert_volume_mappings_to_singularity_format,
        mock_handle_device_requests,
        mock_build_command_args,
        mock_get_volume_mappings_mlcube,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
        mock_logger,
    ):
        mock_handle_device_requests.return_value = [MagicMock()]  # GPU requested
        mock_build_command_args.return_value = ["--data_path=/mlcube_io0"]
        mock_ensure_image.return_value = str(
            self.test_dir / "brats_singularity_images" / "brainles_test-image_latest"
        )
        mock_client.run.return_value = iter([])

        captured_env = {}

        def capture_env(*args, **kwargs):
            captured_env["value"] = os.environ.get(
                "SINGULARITYENV_CUDA_VISIBLE_DEVICES"
            )
            return iter([])

        mock_client.run.side_effect = capture_env

        os.environ["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = "3"
        try:
            run_container(
                algorithm=self.algorithm_gpu,
                data_path=self.data_folder,
                output_path=self.output_folder,
                cuda_devices="0",
                force_cpu=False,
            )

            # cuda_devices parameter takes precedence while running
            self.assertEqual(captured_env["value"], "0")
            # pre-existing value is restored afterwards
            self.assertEqual(os.environ["SINGULARITYENV_CUDA_VISIBLE_DEVICES"], "3")
            # overriding a user-provided value is flagged
            warnings = [call.args[0] for call in mock_logger.warning.call_args_list]
            self.assertTrue(
                any("already set" in message for message in warnings),
                f"Expected an 'already set' warning, got: {warnings}",
            )
        finally:
            os.environ.pop("SINGULARITYENV_CUDA_VISIBLE_DEVICES", None)

    @patch("brats.core.singularity.logger")
    @patch("brats.core.singularity._log_algorithm_info")
    @patch("brats.core.singularity._ensure_image")
    @patch("brats.core.singularity._get_additional_files_path")
    @patch("brats.core.singularity._get_volume_mappings_mlcube")
    @patch("brats.core.singularity._build_command_args")
    @patch("brats.core.singularity._handle_device_requests")
    @patch("brats.core.singularity._convert_volume_mappings_to_singularity_format")
    @patch("brats.core.singularity.Client")
    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity._get_docker_working_dir")
    def test_run_container_cpu_does_not_touch_cuda_env(
        self,
        mock_get_docker_working_dir,
        mock_subprocess_run,
        mock_client,
        mock_convert_volume_mappings_to_singularity_format,
        mock_handle_device_requests,
        mock_build_command_args,
        mock_get_volume_mappings_mlcube,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
        mock_logger,
    ):
        mock_handle_device_requests.return_value = []  # CPU run
        mock_build_command_args.return_value = ["--data_path=/mlcube_io0"]
        mock_ensure_image.return_value = str(
            self.test_dir / "brats_singularity_images" / "brainles_test-image_latest"
        )

        captured_env = {}

        def capture_env(*args, **kwargs):
            captured_env["value"] = os.environ.get(
                "SINGULARITYENV_CUDA_VISIBLE_DEVICES"
            )
            return iter([])

        mock_client.run.side_effect = capture_env

        run_container(
            algorithm=self.algorithm_gpu,
            data_path=self.data_folder,
            output_path=self.output_folder,
            cuda_devices="0",
            force_cpu=False,
        )

        self.assertIsNone(captured_env["value"])
        options = mock_client.run.call_args.kwargs["options"]
        self.assertNotIn("--nv", options)
        self.assertNotIn("SINGULARITYENV_CUDA_VISIBLE_DEVICES", os.environ)
