import shutil

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from brats.core.singularity import (
    run_container,
    _ensure_image,
    _get_docker_working_dir,
    _build_command_args,
    _convert_volume_mappings_to_singularity_format,
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
    def test_ensure_image_pulls_if_missing(self, MockExists, MockLogger, MockPull):
        # Arrange: simulate missing image file
        MockExists.return_value = False

        fake_image_path = "/tmp/brats_singularity_images/test-image"
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
        image = "brainles/test-image:latest"
        MockEnsureImage.return_value = image
        MockDockerClient.images.get.return_value = MagicMock(
            attrs={"Config": {"WorkingDir": "/workspace"}}
        )

        working_dir = _get_docker_working_dir(image)
        self.assertEqual(working_dir, Path("/workspace"))

    @patch("brats.core.singularity.subprocess.run")
    @patch("brats.core.singularity.Path.exists")
    def test_ensure_image_returns_if_exists(self, MockExists, MockPull):
        # Arrange: simulate existing image file
        MockExists.return_value = True

        fake_image_path = "/tmp/brats_singularity_images/fake_image"
        fake_puller = iter([])
        MockPull.return_value = (fake_image_path, fake_puller)

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

    def test_convert_volume_mappings_to_singularity_format(self):
        result = _convert_volume_mappings_to_singularity_format(
            volume_mappings={
                str(self.data_folder.absolute()): {"bind": "/input", "mode": "rw"},
                str(self.output_folder.absolute()): {"bind": "/output", "mode": "rw"},
            }
        )
        expected = [
            f"{str(self.data_folder.absolute())}:/input",
            f"{str(self.output_folder.absolute())}:/output",
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
