import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil

from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import OUTPUT_NAME_SCHEMA


class TestBraTSAlgorithm(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_folder = self.test_dir / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder = self.test_dir / "output"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.subject_A_folder = self.data_folder / "A"
        self.subject_A_folder.mkdir(parents=True, exist_ok=True)
        # Create mock file paths
        self.input_files = {
            "t1c": self.subject_A_folder / "A-t1c.nii.gz",
            "t1n": self.subject_A_folder / "A-t1n.nii.gz",
            "t2f": self.subject_A_folder / "A-t2f.nii.gz",
            "t2w": self.subject_A_folder / "A-t2w.nii.gz",
        }
        for file in self.input_files.values():
            file.touch()

        # the core inference method is the same for all segmentation and inpainting algorithms, we use AdultGliomaSegmenter as an example during testing
        self.segmenter = AdultGliomaPostTreatmentSegmenter()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch("brats.core.brats_algorithm.run_container")
    @patch("brats.core.segmentation_algorithms.input_sanity_check")
    @patch("brats.core.brats_algorithm.InferenceSetup")
    def test_infer_single(
        self, mock_inference_setup, mock_input_sanity_check, mock_run_container
    ):

        # Mock InferenceSetup context manager
        mock_inference_setup_ret = mock_inference_setup.return_value
        mock_inference_setup_ret.__enter__.return_value = (
            self.data_folder,
            self.output_folder,
        )

        def create_output_file(*args, **kwargs):
            subject_id = self.segmenter.algorithm.run_args.input_name_schema.format(
                id=0
            )
            alg_output_file = self.output_folder / OUTPUT_NAME_SCHEMA[
                self.segmenter.task
            ].format(subject_id=subject_id)
            alg_output_file.touch()

        mock_run_container.side_effect = create_output_file

        output_file = self.output_folder / "output.nii.gz"
        self.segmenter.infer_single(
            t1c=self.input_files["t1c"],
            t1n=self.input_files["t1n"],
            t2f=self.input_files["t2f"],
            t2w=self.input_files["t2w"],
            output_file=output_file,
        )
        mock_input_sanity_check.assert_called_once()
        mock_run_container.assert_called_once()

        self.assertTrue(output_file.exists())

    @patch("brats.core.brats_algorithm.run_container")
    @patch("brats.core.segmentation_algorithms.input_sanity_check")
    @patch("brats.core.brats_algorithm.InferenceSetup")
    def test_infer_batch(
        self, mock_inference_setup, mock_input_sanity_check, mock_run_container
    ):

        # Mock InferenceSetup context manager
        mock_inference_setup_ret = mock_inference_setup.return_value
        mock_inference_setup_ret.__enter__.return_value = (
            self.data_folder,
            self.output_folder,
        )

        def create_output_file(*args, **kwargs):
            subject_id = self.segmenter.algorithm.run_args.input_name_schema.format(
                id=0
            )
            alg_output_file = self.output_folder / OUTPUT_NAME_SCHEMA[
                self.segmenter.task
            ].format(subject_id=subject_id)
            alg_output_file.touch()

        mock_run_container.side_effect = create_output_file

        self.segmenter.infer_batch(
            data_folder=self.data_folder, output_folder=self.output_folder
        )
        mock_input_sanity_check.assert_called_once()
        mock_run_container.assert_called_once()
        output_file = self.output_folder / "A.nii.gz"
        self.assertTrue(output_file.exists())
