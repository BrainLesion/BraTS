from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from brats.constants import ADDITIONAL_FILES_FOLDER, ZENODO_RECORD_BASE_URL
from brats.utils.exceptions import ZenodoException


def get_dummy_path() -> Path:
    dummy = ADDITIONAL_FILES_FOLDER / "dummy"
    dummy.mkdir(exist_ok=True, parents=True)
    return dummy


def check_additional_files_path(record_id: str) -> Path:
    """Check if latest additional files are present and download them otherwise.

    Args:
        record_id (str): Zenodo record ID.

    Raises:
        ZenodoException: If the additional files are not found locally and Zenodo is unreachable.

    Returns:
        Path: Path to the additional files folder.
    """

    try:
        zenodo_response = _get_zenodo_metadata_and_archive_url(record_id=record_id)
    except ZenodoException as e:
        # A record with no downloadable archive (or an unexpected API error) must
        # not abort the run when a valid local cache already exists.
        logger.warning(f"Could not resolve a Zenodo archive for {record_id}: {e}")
        zenodo_response = None

    record_additional_files_pattern = f"{record_id}_v*.*.*"
    matching_folders = list(
        ADDITIONAL_FILES_FOLDER.glob(record_additional_files_pattern)
    )
    # Get the latest downloaded additional_files
    latest_downloaded_additional_files = _get_latest_version_folder_name(
        matching_folders
    )

    if not latest_downloaded_additional_files:
        if not zenodo_response:
            msg = "Additional files not found locally and Zenodo could not be reached. Exiting..."
            logger.error(msg)
            raise ZenodoException(msg)
        logger.info("Additional files not found locally")

        zenodo_metadata, archive_urls = zenodo_response
        return _download_additional_files(
            zenodo_metadata=zenodo_metadata,
            record_id=record_id,
            archive_urls=archive_urls,
        )

    logger.info(
        f"Found downloaded local additional_files: {latest_downloaded_additional_files}"
    )

    if not zenodo_response:
        logger.warning(
            "Zenodo server could not be reached. Using the latest downloaded additional files."
        )
        return ADDITIONAL_FILES_FOLDER / latest_downloaded_additional_files

    zenodo_metadata, archive_urls = zenodo_response
    # Compare the latest downloaded additional files with the latest Zenodo version
    if zenodo_metadata["version"] == latest_downloaded_additional_files.split("_v")[1]:
        logger.info(
            f"Latest additional files ({latest_downloaded_additional_files}) are already present."
        )
        return ADDITIONAL_FILES_FOLDER / latest_downloaded_additional_files

    logger.info(
        f"New additional files available on Zenodo "
        f"({zenodo_metadata['version']}). "
        f"Deleting old and fetching new additional files..."
    )
    # delete old additional files
    shutil.rmtree(
        ADDITIONAL_FILES_FOLDER / latest_downloaded_additional_files,
        onerror=lambda func, path, excinfo: logger.warning(
            f"Failed to delete {path}: {excinfo}"
        ),
    )
    return _download_additional_files(
        zenodo_metadata=zenodo_metadata,
        record_id=record_id,
        archive_urls=archive_urls,
    )


def _get_latest_version_folder_name(folders: list[Path]) -> str | None:
    """Get the latest (non empty) version folder name from the list of folders.

    Args:
        folders (List[Path]): List of folders matching the pattern.

    Returns:
        str | None: Latest version folder name if one exists, else None.
    """
    if not folders:
        return None
    latest_downloaded_folder = max(
        folders, key=lambda x: tuple(map(int, x.name.split("_v")[1].split(".")))
    )
    # check folder is not empty
    if not list(latest_downloaded_folder.glob("*")):
        return None
    return latest_downloaded_folder.name


def _select_archive_urls(data: dict[str, Any], record_id: str) -> list[str]:
    """Return the download URL(s) for a record's additional-files archive.

    Zenodo's InvenioRDM API exposes files as ``files[].links.self`` (a direct
    content URL); older responses also had ``links.archive``. Prefer the legacy
    archive link when present, otherwise use the record's ``.zip`` file(s).

    Only ``.zip`` archives are supported: the download path extracts every URL
    with :mod:`zipfile`, so a non-archive file would fail with ``BadZipFile``.

    Args:
        data (dict): Parsed Zenodo record JSON.
        record_id (str): Zenodo record ID (for error messages).

    Returns:
        list[str]: One or more archive download URLs.

    Raises:
        ZenodoException: If the record exposes no downloadable ``.zip`` archive.
    """
    links = data.get("links") or {}
    archive = links.get("archive")
    if isinstance(archive, str) and archive:
        return [archive]

    files = data.get("files") or []
    zips = [
        file
        for file in files
        if str(file.get("key", "")).lower().endswith(".zip")
        and (file.get("links") or {}).get("self")
    ]
    if not zips:
        raise ZenodoException(
            f"Zenodo record {record_id} exposes no downloadable .zip archive"
        )
    return [file["links"]["self"] for file in zips]


def _get_zenodo_metadata_and_archive_url(
    record_id: str,
) -> tuple[dict[str, Any], list[str]] | None:
    """Get the metadata for the Zenodo record and the archive download url(s).

    Args:
        record_id (str): Zenodo record ID.

    Returns:
        Union[Tuple[Dict,List[str]], None]: Tuple with the metadata dict and the
            archive download URL(s) if the record exists, else None.
    """
    try:
        response = requests.get(f"{ZENODO_RECORD_BASE_URL}/{record_id}")
        if response.status_code != 200:
            error_msg = (
                f"Can not find additional files for record_id '{record_id}' "
                f"on Zenodo ({response.status_code=}). Exiting..."
            )
            logger.error(error_msg)
            raise ZenodoException(error_msg)

        data = response.json()
        return data["metadata"], _select_archive_urls(data, record_id)

    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to fetch Zenodo metadata: {e}")
        return None


def _download_additional_files(
    zenodo_metadata: dict[str, object],
    record_id: str,
    archive_urls: str | list[str],
) -> Path:
    """Download the latest additional files from Zenodo for the requested
    record and extract them to the target folder.

    Args:
        zenodo_metadata (Dict): Metadata for the Zenodo record.
        record_id (str): Zenodo record ID.
        archive_urls (Union[str, List[str]]): URL(s) to the archive file(s).

    Raises:
        ZenodoException: If the download fails or the response is not successful.

    Returns:
       Path: Path to the folder where the additional files were extracted.
    """
    urls = [archive_urls] if isinstance(archive_urls, str) else list(archive_urls)
    record_folder = (
        ADDITIONAL_FILES_FOLDER / f"{record_id}_v{zenodo_metadata['version']}"
    )
    # ensure folder exists
    record_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading additional files from Zenodo. This might take a while...")
    try:
        for archive_url in urls:
            # Make a GET request to the URL
            response = requests.get(archive_url, stream=True)
            # Ensure the request was successful
            if response.status_code != 200:
                error_msg = f"Failed to download additional files. Status code: {response.status_code}"
                logger.error(error_msg)
                raise ZenodoException(error_msg)

            _extract_archive(response=response, record_folder=record_folder)
    except Exception:
        # A partially populated version folder would be mistaken for a complete
        # cache by _get_latest_version_folder_name, so drop it on any failure.
        shutil.rmtree(record_folder, ignore_errors=True)
        raise

    logger.info(f"Zip file extracted successfully to {record_folder}")
    return record_folder


def _extract_archive(response: requests.Response, record_folder: Path) -> None:
    # Stream the archive to disk instead of buffering it in memory: some
    # additional-files records are several GB and would otherwise spike host RAM.
    # The temp archive lives next to (not inside) the version folder so that an
    # in-progress or abruptly terminated download cannot be mistaken for a
    # complete cache by _get_latest_version_folder_name's glob("*").
    record_folder = Path(record_folder)
    chunk_size = 1024  # 1KB
    fd, tmp_name = tempfile.mkstemp(
        dir=record_folder.parent, prefix=".download-", suffix=".zip.part"
    )
    tmp_path = Path(tmp_name)
    try:
        with (
            os.fdopen(fd, "wb") as tmp_file,
            Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Downloading additional files..."),
                TextColumn("[cyan]{task.completed:.2f} MB"),
                transient=True,
            ) as progress,
        ):
            task = progress.add_task("", total=None)  # Indeterminate progress
            for data in response.iter_content(chunk_size=chunk_size):
                tmp_file.write(data)
                progress.update(
                    task, advance=len(data) / (chunk_size**2)
                )  # Convert bytes to MB

        # Extract the downloaded zip file to the target folder
        with zipfile.ZipFile(tmp_path) as zip_ref:
            zip_ref.extractall(record_folder)
    finally:
        tmp_path.unlink(missing_ok=True)

    # check if the extracted file is still a zip
    for f in record_folder.iterdir():
        if f.is_file() and f.suffix == ".zip":
            with zipfile.ZipFile(f) as zip_ref:
                files = zip_ref.namelist()
                with Progress(transient=True) as progress:
                    task = progress.add_task(
                        "[cyan]Extracting files...", total=len(files)
                    )
                    # Iterate over the files and extract them
                    for i, file in enumerate(files):
                        zip_ref.extract(file, record_folder)
                        # Update the progress bar
                        progress.update(task, completed=i + 1)
            f.unlink()  # remove zip after extraction
