import shutil
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock, patch

from loguru import logger

from brats.utils.data_handling import (
    InferenceSetup,
    add_log_file_handler,
    input_sanity_check,
    remove_tmp_folder,
)
from brats.utils.logging import enable


class TestDataHandlingUtils(unittest.TestCase):
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

    def test_inference_setup_with_log_file(self):
        # Create a temporary log file
        tmp_log_file = Path(tempfile.mktemp())

        with InferenceSetup(log_file=tmp_log_file) as (
            tmp_data_folder,
            tmp_output_folder,
        ):
            # Check that the folders are created
            self.assertTrue(tmp_data_folder.is_dir())
            self.assertTrue(tmp_output_folder.is_dir())

            # Check that the log file exists
            self.assertTrue(tmp_log_file.exists())

        # Check if folders are cleaned up
        self.assertFalse(tmp_data_folder.exists())
        self.assertFalse(tmp_output_folder.exists())

        # Remove the temporary log file
        tmp_log_file.unlink(missing_ok=True)

    def test_inference_setup_without_log_file(self):
        with InferenceSetup() as (tmp_data_folder, tmp_output_folder):
            # Check that the folders are created
            self.assertTrue(tmp_data_folder.is_dir())
            self.assertTrue(tmp_output_folder.is_dir())

        # Check if folders are cleaned up
        self.assertFalse(tmp_data_folder.exists())
        self.assertFalse(tmp_output_folder.exists())

    def test_inference_setup_success_cleans_up_temp_log(self):
        with patch.object(tempfile, "tempdir", str(self.test_dir)):
            before = list(self.test_dir.glob("brats_*.log"))
            with InferenceSetup():
                pass
            after = list(self.test_dir.glob("brats_*.log"))
            self.assertEqual(len(after), len(before))

    def test_inference_setup_error_preserves_temp_log(self):
        with patch.object(tempfile, "tempdir", str(self.test_dir)):
            before = list(self.test_dir.glob("brats_*.log"))
            with self.assertRaises(RuntimeError), InferenceSetup():
                raise RuntimeError("test error")
            after = list(self.test_dir.glob("brats_*.log"))
            self.assertEqual(len(after), len(before) + 1)
            # Clean up the preserved log
            for log in after:
                if log not in before:
                    log.unlink()

    def test_inference_setup_with_log_file_on_error(self):
        tmp_log_file = self.test_dir / "error.log"
        enable()
        with self.assertRaises(RuntimeError):  # noqa: SIM117
            with InferenceSetup(log_file=tmp_log_file) as (
                _tmp_data_folder,
                _tmp_output_folder,
            ):
                raise RuntimeError("test error")
        self.assertTrue(tmp_log_file.exists())
        self.assertGreater(tmp_log_file.stat().st_size, 0)
        tmp_log_file.unlink()

    def test_remove_tmp_folder_success(self):
        # Test successful removal of a folder
        temp_folder = Path(tempfile.mkdtemp())
        remove_tmp_folder(temp_folder)
        self.assertFalse(temp_folder.exists())

    def test_remove_tmp_folder_permission_error(self):
        # Test handling of PermissionError
        # Create a folder and then set it to read-only to simulate a permission error
        temp_folder = Path(tempfile.mkdtemp())
        temp_folder.chmod(0o444)  # Read-only permissions
        with suppress(PermissionError):
            remove_tmp_folder(temp_folder)
        self.assertFalse(temp_folder.exists())  # Folder should still be removed

    def test_remove_tmp_folder_file_not_found(self):
        # Test handling of FileNotFoundError
        fake_folder = Path(self.test_dir / "non_existent_folder")
        # Ensure the folder does not exist
        self.assertFalse(fake_folder.exists())
        remove_tmp_folder(fake_folder)
        # No assertion needed as the function should handle the error internally

    def test_add_log_file_handler(self):
        # Test adding a log file handler
        log_file = Path(tempfile.mktemp())
        handler_id = add_log_file_handler(log_file)
        self.assertGreater(handler_id, 0)  # Ensure a positive handler ID is returned

        # Check that the log file exists and is writable
        self.assertTrue(log_file.exists())

        # Clean up
        logger.remove(handler_id)
        log_file.unlink(missing_ok=True)

    @patch("brats.utils.data_handling.nib.load")
    @patch("brats.utils.data_handling.logger.warning")
    def test_input_sanity_check_correct_shape(self, mock_warning, mock_nib_load):
        # Mock nib.load to return an object with shape (240, 240, 155)
        mock_img = MagicMock()
        mock_img.shape = (240, 240, 155)
        mock_nib_load.return_value = mock_img

        # Call the function with correct shapes
        input_sanity_check("t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz")

        # Ensure no warnings are logged
        mock_warning.assert_not_called()

    @patch("brats.utils.data_handling.nib.load")
    @patch("brats.utils.data_handling.logger.warning")
    def test_input_sanity_check_incorrect_shape(self, mock_warning, mock_nib_load):
        # Mock nib.load to return an object with shape (191, 512, 512) for one image
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
