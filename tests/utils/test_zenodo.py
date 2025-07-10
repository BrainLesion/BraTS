import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path
from io import BytesIO
import requests

# Import the module that contains the functions
from brats.constants import ADDITIONAL_FILES_FOLDER
from brats.utils.exceptions import ZenodoException
from brats.utils.zenodo import (
    _extract_archive,
    check_additional_files_path,
    _get_latest_version_folder_name,
    _get_zenodo_metadata_and_archive_url,
    _download_additional_files,
)


class TestZenodoUtils(unittest.TestCase):

    @patch("brats.utils.zenodo._get_zenodo_metadata_and_archive_url")
    @patch("brats.utils.zenodo._get_latest_version_folder_name")
    @patch("brats.utils.zenodo._download_additional_files")
    @patch("brats.utils.zenodo.shutil.rmtree")
    @patch("brats.utils.zenodo.Path.mkdir")
    @patch("brats.utils.zenodo.Path.glob")
    def test_check_additional_files_path(
        self,
        mock_glob,
        mock_mkdir,
        mock_rmtree,
        mock_download_additional_files,
        mock_get_latest_version,
        mock_get_zenodo_metadata,
    ):
        # Setup
        mock_record_id = "12345"
        mock_matching_folder = MagicMock(spec=Path)
        mock_matching_folder.name = f"{mock_record_id}_v1.0.0"
        mock_glob.return_value = [mock_matching_folder]
        mock_get_latest_version.return_value = f"{mock_record_id}_v1.0.0"
        mock_get_zenodo_metadata.return_value = (
            {"version": "1.0.0"},
            "http://test.url",
        )

        # Test when local additional_files are up-to-date
        result = check_additional_files_path(mock_record_id)
        self.assertEqual(result, ADDITIONAL_FILES_FOLDER / f"{mock_record_id}_v1.0.0")
        mock_rmtree.assert_not_called()
        mock_download_additional_files.assert_not_called()

        # Test when new additional_files are available
        mock_get_zenodo_metadata.return_value = (
            {"version": "2.0.0"},
            "http://test.url",
        )
        result = check_additional_files_path(mock_record_id)
        mock_rmtree.assert_called_once()
        mock_download_additional_files.assert_called_once()

    @patch("brats.utils.zenodo._get_zenodo_metadata_and_archive_url")
    @patch("brats.utils.zenodo._get_latest_version_folder_name")
    @patch("brats.utils.zenodo._download_additional_files")
    @patch("brats.utils.zenodo.shutil.rmtree")
    @patch("brats.utils.zenodo.Path.mkdir")
    @patch("brats.utils.zenodo.Path.glob")
    def test_check_additional_files_path_not_present_zenodo_unreachable(
        self,
        mock_glob,
        mock_mkdir,
        mock_rmtree,
        mock_download_additional_files,
        mock_get_latest_version,
        mock_get_zenodo_metadata,
    ):
        # Setup
        mock_record_id = "12345"
        mock_matching_folder = MagicMock(spec=Path)
        mock_matching_folder.name = f"{mock_record_id}_v1.0.0"
        mock_glob.return_value = [mock_matching_folder]
        mock_get_latest_version.return_value = None
        mock_get_zenodo_metadata.return_value = None

        # Test when local additional_files are not present and Zenodo is unreachable
        with self.assertRaises(ZenodoException):

            check_additional_files_path(mock_record_id)
            mock_rmtree.assert_not_called()
            mock_download_additional_files.assert_not_called()

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url(self, mock_get):
        # Setup
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": {"version": "1.0.0"},
            "links": {"archive": "http://test.url"},
        }
        mock_get.return_value = mock_response

        zenodo_response = _get_zenodo_metadata_and_archive_url("12345")
        if not zenodo_response:
            self.fail("Expected a valid response, but got None")
        metadata, archive_url = zenodo_response
        self.assertEqual(metadata, {"version": "1.0.0"})
        self.assertEqual(archive_url, "http://test.url")

        # Test when the request fails
        mock_get.side_effect = requests.exceptions.RequestException("Failed")
        ret = _get_zenodo_metadata_and_archive_url("12345")
        self.assertIsNone(ret)

    @patch("brats.utils.zenodo.ADDITIONAL_FILES_FOLDER", Path(tempfile.mkdtemp()))
    @patch("brats.utils.zenodo._extract_archive")
    @patch("brats.utils.zenodo.requests.get")
    def test_download_additional_files(
        self,
        mock_requests_get,
        mock_extract_archive,
    ):
        # Setup
        mock_zenodo_metadata = {"version": "1.0.0"}
        mock_archive_url = "http://test.url"
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=[b"data"])
        mock_requests_get.return_value = mock_response

        # Call the function
        result_path = _download_additional_files(
            mock_zenodo_metadata, "12345", mock_archive_url
        )

        # Assertions
        # mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_requests_get.assert_called_once_with(mock_archive_url, stream=True)
        # mock_zipfile_instance.extractall.assert_called_once_with(result_path)
        mock_extract_archive.assert_called_once()

    @patch("brats.utils.zenodo.ADDITIONAL_FILES_FOLDER", Path(tempfile.mkdtemp()))
    @patch("brats.utils.zenodo._extract_archive")
    @patch("brats.utils.zenodo.requests.get")
    def test_download_additional_files_zenodo_error(
        self,
        mock_requests_get,
        mock_extract_archive,
    ):
        # Setup
        mock_zenodo_metadata = {"version": "1.0.0"}
        mock_archive_url = "http://test.url"
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 400
        mock_response.iter_content = MagicMock(return_value=[b"data"])
        mock_requests_get.return_value = mock_response

        # Call the function
        with self.assertRaises(ZenodoException):
            _download_additional_files(mock_zenodo_metadata, "12345", mock_archive_url)

        # Assertions
        # mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_requests_get.assert_called_once_with(mock_archive_url, stream=True)
        mock_extract_archive.assert_not_called()

    @patch("brats.utils.zenodo.zipfile.ZipFile")
    @patch("brats.utils.zenodo.BytesIO", new_callable=MagicMock)
    @patch("brats.utils.zenodo.Progress")
    def test_extract_archive(self, mock_progress, mock_bytes_io, mock_zipfile):
        # Setup
        mock_response = MagicMock(spec=requests.Response)
        mock_response.iter_content.return_value = [b"data"]
        mock_record_folder = MagicMock(spec=Path)

        mock_bytes_io_instance = MagicMock(spec=BytesIO)
        mock_bytes_io.return_value = mock_bytes_io_instance  # Mock the instantiation

        mock_zipfile_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zipfile_instance

        # Call the function
        _extract_archive(mock_response, mock_record_folder)

    def test_get_latest_version_folder_name(self):
        # Test case when folders are provided
        folder1 = MagicMock(spec=Path)
        folder1.name = "12345_v1.0.0"
        folder1.__str__.return_value = "12345_v1.0.0"
        folder2 = MagicMock(spec=Path)
        folder2.name = "12345_v2.0.0"
        folder2.__str__.return_value = "12345_v2.0.0"
        folder3 = MagicMock(spec=Path)
        folder3.name = "12345_v1.5.0"
        folder3.__str__.return_value = "12345_v1.5.0"

        folder2.glob.return_value = ["not empty"]
        folder1.glob.return_value = ["not empty"]
        folder3.glob.return_value = []

        result = _get_latest_version_folder_name([folder1, folder2, folder3])
        self.assertEqual(result, "12345_v2.0.0")

        # Test case when no folders are provided
        result = _get_latest_version_folder_name([])
        self.assertIsNone(result)

        # Test case when folder is empty
        folder2.glob.return_value = []
        result = _get_latest_version_folder_name([folder1, folder2])
        self.assertIsNone(result)
