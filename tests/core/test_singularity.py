import shutil

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from brats.core.singularity import run_container, _ensure_image
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

    @patch("brats.core.singularity.Client.pull")
    @patch("brats.core.singularity.logger")
    @patch("brats.core.singularity.Path.exists")
    def test_ensure_image_pulls_if_missing(self, MockExists, MockLogger, MockPull):
        # Arrange: simulate missing image file
        MockExists.return_value = False

        fake_image_path = "/tmp/fake_image.sif"
        fake_puller = iter([
            "singularity pull --name /tmp/test-image:latest.sif docker://test-image:latest"
        ])
        MockPull.return_value = (fake_image_path, fake_puller)


        result = _ensure_image("test-image:latest")

        # Assert
        MockPull.assert_called_once_with(
            "docker://test-image:latest", stream=True, pull_folder="/tmp"
        )
        assert result == fake_image_path
        MockLogger.info.assert_any_call(f"Pulling Singularity image {fake_image_path}")
        MockLogger.info.assert_any_call("singularity pull --name /tmp/test-image:latest.sif docker://test-image:latest")


    @patch("brats.core.singularity.Client.pull")
    @patch("brats.core.singularity.Path.exists")
    def test_ensure_image_returns_if_exists(self, MockExists, MockPull):
        # Arrange: simulate existing image file
        MockExists.return_value = True

        fake_image_path = "/tmp/fake_image.sif"
        fake_puller = iter([])
        MockPull.return_value = (fake_image_path, fake_puller)

        from brats.core.singularity import _ensure_image
        result = _ensure_image("already-there:1.0")

        # Assert
        assert result == fake_image_path
        MockPull.assert_called_once()
        # puller should not be consumed since image exists

        
    @patch("brats.core.singularity._log_algorithm_info")
    @patch("brats.core.singularity._ensure_image")
    @patch("brats.core.singularity._get_additional_files_path")
    @patch("brats.core.singularity._get_volume_mappings")
    @patch("brats.core.singularity._build_args")
    @patch("brats.core.singularity._handle_device_requests")
    @patch("brats.core.singularity._convert_volume_mappings_to_singularity_format")
    @patch("brats.core.singularity.Client")
    def test_run_singularity_container(
        self,
        mock_client,
        mock_convert_volume_mappings_to_singularity_format,
        mock_handle_device_requests,
        mock_build_args,
        mock_get_volume_mappings,
        mock_get_additional_files_path,
        mock_ensure_image,
        mock_log_algorithm_info,
    ):

        # setup mocks
        mock_build_args.return_value = ("args", {})

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
        mock_get_volume_mappings.assert_called_once()
        mock_build_args.assert_called_once()
        mock_handle_device_requests.assert_called_once()
        mock_convert_volume_mappings_to_singularity_format.assert_called_once()
