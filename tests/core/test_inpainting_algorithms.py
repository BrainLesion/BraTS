import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from brats import Inpainter
from brats.constants import InpaintingAlgorithms


class TestInpaintingAlgorithms(unittest.TestCase):
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
        self.t1n = self.data_folder / "subject-t1n-voided.nii.gz"
        self.mask = self.data_folder / "subject-mask.nii.gz"
        # Create dummy files
        for img in [self.t1n, self.mask]:
            img.touch(exist_ok=True)

        self.segmenter = Inpainter()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    ### Standardization tests

    @patch("brats.core.inpainting_algorithms.input_sanity_check")
    def test_successful_single_standardization(self, mock_input_sanity_check):
        subject_id = "test_subject"
        self.segmenter._standardize_single_inputs(
            data_folder=self.tmp_data_folder,
            subject_id=subject_id,
            inputs={
                "t1n": self.t1n,
                "mask": self.mask,
            },
        )
        subject_folder = self.tmp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        self.assertTrue((subject_folder / f"{subject_id}-t1n-voided.nii.gz").exists())
        self.assertTrue((subject_folder / f"{subject_id}-mask.nii.gz").exists())

    @patch("brats.core.inpainting_algorithms.input_sanity_check")
    @patch("sys.exit")
    @patch.object(logger, "error")
    def test_single_standardize_handle_file_not_found_error(
        self, mock_logger, mock_exit, mock_input_sanity_check
    ):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1n = "non_existent_file.nii.gz"
        self.segmenter._standardize_single_inputs(
            data_folder=self.data_folder,
            subject_id=subject_id,
            inputs={
                "t1n": t1n,
                "mask": self.mask,
            },
        )
        mock_logger.assert_called()
        mock_exit.assert_called_with(1)

    @patch("brats.core.inpainting_algorithms.Inpainter._standardize_single_inputs")
    def test_standardize_segmentation_inputs_list(self, mock_standardize_single_inputs):
        subjects = [f for f in self.data_folder.iterdir() if f.is_dir()]
        mapping = self.segmenter._standardize_batch_inputs(
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

    ### Initialization tests

    def test_inpainter_initialization(self):
        # Test default initialization
        inpainter = Inpainter()
        self.assertIsInstance(inpainter, Inpainter)

        # Test with custom arguments
        custom_inpainter = Inpainter(
            algorithm=InpaintingAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_inpainter, Inpainter)
