import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

from brats.utils import (
    standardize_segmentation_inputs,
    standardize_segmentation_inputs_list,
    input_sanity_check,
)


class TestStandardizeInputs(unittest.TestCase):
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
        self.t1c = self.data_folder / "subject-t1c.nii.gz"
        self.t1n = self.data_folder / "subject-t1n.nii.gz"
        self.t2f = self.data_folder / "subject-t2f.nii.gz"
        self.t2w = self.data_folder / "subject-t2w.nii.gz"
        # Create dummy files
        for img in [self.t1c, self.t1n, self.t2f, self.t2w]:
            img.touch(exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    @patch("brats.utils.input_sanity_check")
    def test_successful_standardization(self, mock_input_sanity_check):
        subject_id = "test_subject"
        standardize_segmentation_inputs(
            data_folder=self.tmp_data_folder,
            subject_id=subject_id,
            t1c=self.t1c,
            t1n=self.t1n,
            t2f=self.t2f,
            t2w=self.t2w,
        )
        subject_folder = self.tmp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        for img_type in ["t1c", "t1n", "t2f", "t2w"]:
            self.assertTrue(
                (subject_folder / f"{subject_id}-{img_type}.nii.gz").exists()
            )

    @patch("brats.utils.input_sanity_check")
    @patch("sys.exit")
    @patch.object(logger, "error")
    def test_handle_file_not_found_error(
        self, mock_logger, mock_exit, mock_input_sanity_check
    ):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1c = "non_existent_file.nii.gz"
        standardize_segmentation_inputs(
            data_folder=self.data_folder,
            subject_id=subject_id,
            t1c=t1c,
            t1n=self.t1n,
            t2f=self.t2f,
            t2w=self.t2w,
        )
        mock_logger.assert_called()
        mock_exit.assert_called_with(1)

    @patch("brats.utils.standardize_segmentation_inputs")
    def test_standardize_segmentation_inputs_list(
        self, mock_standardize_segmentation_inputs
    ):
        subjects = [f for f in self.data_folder.iterdir() if f.is_dir()]
        mapping = standardize_segmentation_inputs_list(
            subjects=subjects,
            tmp_data_folder=self.tmp_data_folder,
            input_name_schema="BraTS-PED-{id:05d}-000",
        )
        self.assertDictEqual(
            mapping,
            {
                "BraTS-PED-00000-000": "subject",
            },
        )
        mock_standardize_segmentation_inputs.assert_called_once()

    @patch("brats.utils.nib.load")
    @patch("brats.utils.logger.warning")
    def test_correct_shape(self, mock_warning, mock_nib_load):
        # Mock nib.load to return an object with shape (240, 240, 155)
        mock_img = MagicMock()
        mock_img.shape = (240, 240, 155)
        mock_nib_load.return_value = mock_img

        # Call the function with correct shapes
        input_sanity_check("t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz")

        # Ensure no warnings are logged
        mock_warning.assert_not_called()

    @patch("brats.utils.nib.load")
    @patch("brats.utils.logger.warning")
    def test_incorrect_shape(self, mock_warning, mock_nib_load):
        # Mock nib.load to return an object with shape (240, 240, 100) for one image
        mock_img_correct = MagicMock()
        mock_img_correct.shape = (240, 240, 155)
        mock_img_incorrect = MagicMock()
        mock_img_incorrect.shape = (191, 512, 512)

        def side_effect(arg):
            if arg == "t1c.nii.gz":
                return mock_img_incorrect
            else:
                return mock_img_correct

        mock_nib_load.side_effect = side_effect

        # Call the function with one incorrect shape
        input_sanity_check("t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz")

        # Ensure warnings are logged
        self.assertTrue(mock_warning.called)
