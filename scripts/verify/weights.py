"""Runtime override of the Zenodo additional-files cache directory.

The BraTS library downloads model weights into ``brats/data/additional_files``
(``brats.constants.ADDITIONAL_FILES_FOLDER``) and caches them there. The
verification suite wants every weighted algorithm to exercise a *fresh*
download without using or touching the real cache.

``brats.utils.zenodo`` binds ``ADDITIONAL_FILES_FOLDER`` as a module-level name
and resolves it at call time, so overriding that attribute for the duration of a
run redirects all downloads into a directory we control. No package source is
modified; the override is process-local and restored afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import brats.utils.zenodo as zenodo


@contextmanager
def weights_cache_override(cache_dir: Path) -> Iterator[Path]:
    """Redirect the Zenodo weights cache to ``cache_dir`` for the block.

    Args:
        cache_dir (Path): Directory to use as the additional-files cache

    Yields:
        Path: The overridden cache directory
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    original = zenodo.ADDITIONAL_FILES_FOLDER
    zenodo.ADDITIONAL_FILES_FOLDER = cache_dir
    try:
        yield cache_dir
    finally:
        zenodo.ADDITIONAL_FILES_FOLDER = original
