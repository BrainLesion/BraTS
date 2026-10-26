import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Import the module that contains the functions
from brats.constants import ADDITIONAL_FILES_FOLDER
from brats.utils.exceptions import ZenodoException
from brats.utils.zenodo import (
    _download_additional_files,
    _extract_archive,
    _get_latest_version_folder_name,
    _get_zenodo_metadata_and_archive_url,
    check_additional_files_path,
    read_only_cache,
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
            ["http://test.url"],
        )

        # Test when local additional_files are up-to-date
        result = check_additional_files_path(mock_record_id)
        self.assertEqual(result, ADDITIONAL_FILES_FOLDER / f"{mock_record_id}_v1.0.0")
        mock_rmtree.assert_not_called()
        mock_download_additional_files.assert_not_called()

        # Test when new additional_files are available
        mock_get_zenodo_metadata.return_value = (
            {"version": "2.0.0"},
            ["http://test.url"],
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

    @patch("brats.utils.zenodo._get_zenodo_metadata_and_archive_url")
    @patch("brats.utils.zenodo._get_latest_version_folder_name")
    @patch("brats.utils.zenodo._download_additional_files")
    @patch("brats.utils.zenodo.shutil.rmtree")
    @patch("brats.utils.zenodo.Path.mkdir")
    @patch("brats.utils.zenodo.Path.glob")
    def test_check_additional_files_path_falls_back_to_cache(
        self,
        mock_glob,
        mock_mkdir,
        mock_rmtree,
        mock_download_additional_files,
        mock_get_latest_version,
        mock_get_zenodo_metadata,
    ):
        # A record with no downloadable archive must not abort the run when a
        # valid local cache already exists.
        mock_record_id = "12345"
        mock_matching_folder = MagicMock(spec=Path)
        mock_matching_folder.name = f"{mock_record_id}_v1.0.0"
        mock_glob.return_value = [mock_matching_folder]
        mock_get_latest_version.return_value = f"{mock_record_id}_v1.0.0"
        mock_get_zenodo_metadata.side_effect = ZenodoException("no zip archive")

        result = check_additional_files_path(mock_record_id)

        self.assertEqual(result, ADDITIONAL_FILES_FOLDER / f"{mock_record_id}_v1.0.0")
        mock_download_additional_files.assert_not_called()

    @patch("brats.utils.zenodo._get_zenodo_metadata_and_archive_url")
    @patch("brats.utils.zenodo._get_latest_version_folder_name")
    @patch("brats.utils.zenodo._download_additional_files")
    @patch("brats.utils.zenodo.shutil.rmtree")
    @patch("brats.utils.zenodo.Path.mkdir")
    @patch("brats.utils.zenodo.Path.glob")
    def test_check_additional_files_path_raises_without_cache(
        self,
        mock_glob,
        mock_mkdir,
        mock_rmtree,
        mock_download_additional_files,
        mock_get_latest_version,
        mock_get_zenodo_metadata,
    ):
        # Without a local cache, an unresolvable archive must still raise.
        mock_record_id = "12345"
        mock_glob.return_value = []
        mock_get_latest_version.return_value = None
        mock_get_zenodo_metadata.side_effect = ZenodoException("no zip archive")

        with self.assertRaises(ZenodoException):
            check_additional_files_path(mock_record_id)

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
        metadata, archive_urls = zenodo_response
        self.assertEqual(metadata, {"version": "1.0.0"})
        self.assertEqual(archive_urls, ["http://test.url"])

        # Test when the request fails
        mock_get.side_effect = requests.exceptions.RequestException("Failed")
        ret = _get_zenodo_metadata_and_archive_url("12345")
        self.assertIsNone(ret)

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_invenio_files(self, mock_get):
        # InvenioRDM exposes files via files[].links.self (no links.archive);
        # this is the shape that caused the KeyError in issue #193.
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": {"version": "1.0.1"},
            "links": {"files": "https://zenodo.org/api/records/1/files"},
            "files": [
                {
                    "key": "weights.zip",
                    "links": {"self": "https://zenodo.org/api/records/1/files/a"},
                }
            ],
        }
        mock_get.return_value = mock_response

        metadata, archive_urls = _get_zenodo_metadata_and_archive_url("1")
        self.assertEqual(metadata, {"version": "1.0.1"})
        self.assertEqual(archive_urls, ["https://zenodo.org/api/records/1/files/a"])

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_selects_zip_only(self, mock_get):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": {"version": "1.0.0"},
            "files": [
                {"key": "readme.txt", "links": {"self": "https://z/readme"}},
                {"key": "weights.zip", "links": {"self": "https://z/weights"}},
            ],
        }
        mock_get.return_value = mock_response

        _, archive_urls = _get_zenodo_metadata_and_archive_url("1")
        self.assertEqual(archive_urls, ["https://z/weights"])

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_multiple_zips(self, mock_get):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": {"version": "1.0.0"},
            "files": [
                {"key": "a.zip", "links": {"self": "https://z/a"}},
                {"key": "b.zip", "links": {"self": "https://z/b"}},
            ],
        }
        mock_get.return_value = mock_response

        _, archive_urls = _get_zenodo_metadata_and_archive_url("1")
        self.assertEqual(archive_urls, ["https://z/a", "https://z/b"])

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_no_files(self, mock_get):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"metadata": {"version": "1.0.0"}}
        mock_get.return_value = mock_response

        with self.assertRaises(ZenodoException):
            _get_zenodo_metadata_and_archive_url("1")

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_requires_zip(self, mock_get):
        # A record with only non-archive files must be rejected explicitly,
        # otherwise the download path would fail later with BadZipFile.
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "metadata": {"version": "1.0.0"},
            "files": [
                {"key": "readme.txt", "links": {"self": "https://z/readme"}},
            ],
        }
        mock_get.return_value = mock_response

        with self.assertRaises(ZenodoException):
            _get_zenodo_metadata_and_archive_url("1")

    @patch("brats.utils.zenodo.requests.get")
    def test_get_zenodo_metadata_and_archive_url_zenodo_error(self, mock_get):
        # Setup
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 400

        with self.assertRaises(ZenodoException):
            _get_zenodo_metadata_and_archive_url("12345")

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
        _download_additional_files(mock_zenodo_metadata, "12345", mock_archive_url)

        # Assertions
        mock_requests_get.assert_called_once_with(mock_archive_url, stream=True)
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
        mock_requests_get.assert_called_once_with(mock_archive_url, stream=True)
        mock_extract_archive.assert_not_called()

    @patch("brats.utils.zenodo.ADDITIONAL_FILES_FOLDER", Path(tempfile.mkdtemp()))
    @patch("brats.utils.zenodo._extract_archive")
    @patch("brats.utils.zenodo.requests.get")
    def test_download_additional_files_multiple(
        self,
        mock_requests_get,
        mock_extract_archive,
    ):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        _download_additional_files({"version": "1.0.0"}, "12345", ["u1", "u2"])

        self.assertEqual(mock_extract_archive.call_count, 2)

    @patch("brats.utils.zenodo._extract_archive")
    @patch("brats.utils.zenodo.requests.get")
    def test_download_additional_files_partial_failure_cleans_up(
        self,
        mock_requests_get,
        mock_extract_archive,
    ):
        cache = Path(tempfile.mkdtemp())
        with patch("brats.utils.zenodo.ADDITIONAL_FILES_FOLDER", cache):
            ok = MagicMock(spec=requests.Response)
            ok.status_code = 200
            bad = MagicMock(spec=requests.Response)
            bad.status_code = 500
            mock_requests_get.side_effect = [ok, bad]

            with self.assertRaises(ZenodoException):
                _download_additional_files({"version": "1.0.0"}, "12345", ["u1", "u2"])

            # The partially populated version folder must not survive, otherwise
            # it would be treated as a complete cache on the next invocation.
            self.assertFalse((cache / "12345_v1.0.0").exists())

    @patch("brats.utils.zenodo._extract_archive")
    @patch("brats.utils.zenodo.requests.get")
    def test_download_additional_files_stages_before_publishing(
        self,
        mock_requests_get,
        mock_extract_archive,
    ):
        cache = Path(tempfile.mkdtemp())
        observed = {}

        def fake_get(url, stream=True):
            # While the download streams, nothing matching the cache glob must be
            # visible: an in-progress download must not look like a full cache.
            observed["matching"] = list(cache.glob("12345_v*.*.*"))
            response = MagicMock(spec=requests.Response)
            response.status_code = 200
            return response

        mock_requests_get.side_effect = fake_get
        with patch("brats.utils.zenodo.ADDITIONAL_FILES_FOLDER", cache):
            result = _download_additional_files({"version": "1.0.0"}, "12345", "u1")

        staging = mock_extract_archive.call_args.kwargs["record_folder"]
        self.assertNotEqual(staging, cache / "12345_v1.0.0")
        self.assertTrue(staging.name.startswith(".12345_v1.0.0.staging-"))
        self.assertEqual(observed["matching"], [])
        self.assertEqual(result, cache / "12345_v1.0.0")
        self.assertTrue(result.is_dir())
        # staging folder is published (renamed), not left behind
        self.assertEqual([p for p in cache.iterdir() if "staging" in p.name], [])

    @patch("brats.utils.zenodo._get_zenodo_metadata_and_archive_url")
    @patch("brats.utils.zenodo._get_latest_version_folder_name")
    @patch("brats.utils.zenodo._download_additional_files")
    @patch("brats.utils.zenodo.shutil.rmtree")
    @patch("brats.utils.zenodo.Path.glob")
    def test_read_only_cache_never_evicts(
        self,
        mock_glob,
        mock_rmtree,
        mock_download,
        mock_get_latest_version,
        mock_get_zenodo_metadata,
    ):
        mock_record_id = "12345"
        mock_glob.return_value = [Path(f"{mock_record_id}_v1.0.0")]
        mock_get_latest_version.return_value = f"{mock_record_id}_v1.0.0"
        mock_get_zenodo_metadata.return_value = ({"version": "2.0.0"}, ["u1"])

        with read_only_cache():
            result = check_additional_files_path(mock_record_id)

        self.assertEqual(result, ADDITIONAL_FILES_FOLDER / f"{mock_record_id}_v1.0.0")
        mock_rmtree.assert_not_called()
        mock_download.assert_not_called()

    def test_extract_archive(self):
        # Build a small zip in memory and stream it through the fake response.
        payload_io = BytesIO()
        with zipfile.ZipFile(payload_io, "w") as archive:
            archive.writestr("weights.bin", b"data")
        payload = payload_io.getvalue()

        mock_response = MagicMock(spec=requests.Response)
        mock_response.iter_content.return_value = [payload]

        with tempfile.TemporaryDirectory() as tmp:
            record_folder = Path(tmp) / "12345_v1.0.0"
            record_folder.mkdir()
            _extract_archive(mock_response, record_folder)
            self.assertEqual((record_folder / "weights.bin").read_bytes(), b"data")
            # The temp download must not be left behind, and must live outside the
            # version folder so it cannot be mistaken for a complete cache.
            self.assertEqual(list(record_folder.glob("*.part")), [])
            self.assertEqual(list(record_folder.parent.glob("*.part")), [])

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


@pytest.mark.network
def test_zenodo_invenio_record_reachable():
    """Live check for issue #193: record 14413387 has no ``links.archive``.

    Opt-in with ``pytest -m network``; skips when Zenodo is unreachable.
    """
    record_id = "14413387"
    zenodo_response = _get_zenodo_metadata_and_archive_url(record_id)
    if zenodo_response is None:
        pytest.skip("Zenodo unreachable")

    metadata, archive_urls = zenodo_response
    assert metadata.get("version")
    assert archive_urls, "expected at least one archive URL"
    assert all(url.startswith("http") for url in archive_urls)
