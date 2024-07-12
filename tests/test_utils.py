import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from brats.utils import standardize_subject_inputs, standardize_subjects_inputs_list


class TestStandardizeInputs(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_folder = self.test_dir / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.temp_data_folder = self.test_dir / "tmp_std_data"
        self.temp_data_folder.mkdir(parents=True, exist_ok=True)

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

    def test_successful_standardization(self):
        subject_id = "test_subject"
        standardize_subject_inputs(
            data_folder=self.temp_data_folder,
            subject_id=subject_id,
            t1c=self.t1c,
            t1n=self.t1n,
            t2f=self.t2f,
            t2w=self.t2w,
        )
        subject_folder = self.temp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        for img_type in ["t1c", "t1n", "t2f", "t2w"]:
            self.assertTrue(
                (subject_folder / f"{subject_id}-{img_type}.nii.gz").exists()
            )

    @patch("sys.exit")
    @patch.object(logger, "error")
    def test_handle_file_not_found_error(self, mock_logger, mock_exit):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1c = "non_existent_file.nii.gz"
        standardize_subject_inputs(
            data_folder=self.data_folder,
            subject_id=subject_id,
            t1c=t1c,
            t1n=self.t1n,
            t2f=self.t2f,
            t2w=self.t2w,
        )
        mock_logger.assert_called()
        mock_exit.assert_called_with(1)

    @patch("brats.utils.standardize_subject_inputs")
    def test_standardize_subjects_inputs_list(self, mock_standardize_subject_inputs):
        subjects = [f for f in self.data_folder.iterdir() if f.is_dir()]
        mapping = standardize_subjects_inputs_list(
            subjects=subjects,
            temp_data_folder=self.temp_data_folder,
            input_name_schema="BraTS-PED-{id:05d}-000",
        )
        self.assertDictEqual(
            mapping,
            {
                "BraTS-PED-00000-000": "subject",
            },
        )
        mock_standardize_subject_inputs.assert_called_once()
