import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from brats import MissingMRI
from brats.constants import MissingMRIAlgorithms


class TestMissingMRIAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_folder = self.test_dir / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.tmp_data_folder = self.test_dir / "tmp_std_data"
        self.tmp_data_folder.mkdir(parents=True, exist_ok=True)

        # Create mock paths for input images
        self.subject_folder = self.data_folder / "subject"
        self.subject_folder.mkdir(parents=True, exist_ok=True)
        self.t1n = self.subject_folder / "subject-t1n.nii.gz"
        self.t1c = self.subject_folder / "subject-t1c.nii.gz"
        self.t2w = self.subject_folder / "subject-t2w.nii.gz"
        # Create dummy files
        for img in [self.t1n, self.t1c, self.t2w]:
            img.touch(exist_ok=True)

        self.missing_mri = MissingMRI()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    # Standardization tests

    @patch("brats.core.missing_mri_algorithms.input_sanity_check")
    def test_successful_single_standardization(self, mock_input_sanity_check):
        subject_id = "test_subject"
        self.missing_mri._standardize_single_inputs(
            data_folder=self.tmp_data_folder,
            subject_id=subject_id,
            inputs={
                "t1n": self.t1n,
                "t1c": self.t1c,
                "t2w": self.t2w,
            },
            subject_modality_separator="-",
        )
        subject_folder = self.tmp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        self.assertTrue((subject_folder / f"{subject_id}-t1n.nii.gz").exists())
        self.assertTrue((subject_folder / f"{subject_id}-t1c.nii.gz").exists())
        self.assertTrue((subject_folder / f"{subject_id}-t2w.nii.gz").exists())

    @patch("brats.core.missing_mri_algorithms.input_sanity_check")
    @patch.object(logger, "error")
    def test_single_standardize_handle_file_not_found_error(
        self, mock_logger, mock_input_sanity_check
    ):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1n = "non_existent_file.nii.gz"

        with self.assertRaises(FileNotFoundError):
            self.missing_mri._standardize_single_inputs(
                data_folder=self.data_folder,
                subject_id=subject_id,
                inputs={
                    "t1n": t1n,
                    "t1c": self.t1c,
                    "t2w": self.t2w,
                },
                subject_modality_separator="-",
            )
            mock_logger.assert_called()

    @patch("brats.core.missing_mri_algorithms.MissingMRI._standardize_single_inputs")
    def test_standardize_segmentation_inputs_list(self, mock_standardize_single_inputs):
        subjects = [f for f in self.data_folder.iterdir() if f.is_dir()]
        mapping = self.missing_mri._standardize_batch_inputs(
            data_folder=self.tmp_data_folder,
            subjects=subjects,
            input_name_schema="BraTS-GLI-{id:05d}-000",
        )
        self.assertDictEqual(
            mapping,
            {
                "BraTS-GLI-00000-000": "subject",
            },
        )
        mock_standardize_single_inputs.assert_called_once()

    @patch("brats.core.missing_mri_algorithms.input_sanity_check")
    @patch("brats.core.brats_algorithm.run_docker_container")
    @patch("brats.core.brats_algorithm.InferenceSetup")
    def test_infer_single_raises_file_not_found_for_missing_output(
        self, mock_inference_setup, mock_run_container, mock_input_sanity_check
    ):
        output_folder = self.test_dir / "single_tmp_output"
        output_folder.mkdir()
        mock_inference_setup.return_value.__enter__.return_value = (
            self.tmp_data_folder,
            output_folder,
        )

        with self.assertRaisesRegex(FileNotFoundError, "No output found for subject"):
            self.missing_mri.infer_single(
                t1c=self.t1c,
                t1n=self.t1n,
                t2w=self.t2w,
                output_file=self.test_dir / "output.nii.gz",
            )

        mock_run_container.assert_called_once()
        mock_input_sanity_check.assert_called_once()

    @patch("brats.core.missing_mri_algorithms.input_sanity_check")
    @patch("brats.core.brats_algorithm.run_docker_container")
    @patch("brats.core.brats_algorithm.InferenceSetup")
    def test_infer_batch_raises_file_not_found_for_missing_output(
        self, mock_inference_setup, mock_run_container, mock_input_sanity_check
    ):
        output_folder = self.test_dir / "batch_tmp_output"
        output_folder.mkdir()
        mock_inference_setup.return_value.__enter__.return_value = (
            self.tmp_data_folder,
            output_folder,
        )

        with self.assertRaisesRegex(FileNotFoundError, "No output found for subject"):
            self.missing_mri.infer_batch(
                data_folder=self.data_folder,
                output_folder=self.test_dir / "outputs",
            )

        mock_run_container.assert_called_once()
        mock_input_sanity_check.assert_called_once()

    # Initialization tests

    def test_missing_mri_initialization(self):
        # Test default initialization
        missing_mri = MissingMRI()
        self.assertIsInstance(missing_mri, MissingMRI)

        # Test with custom arguments
        custom_missing_mri = MissingMRI(
            algorithm=MissingMRIAlgorithms.BraTS24_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_missing_mri, MissingMRI)
