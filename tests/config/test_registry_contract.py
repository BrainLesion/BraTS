"""Config-contract tests for the YAML-driven algorithm registry.

These tests are GPU- and container-less: they validate the integrity of the
algorithm registry (enums, metadata, parameter files and name schemas) and can
run in the regular pull-request CI. Checks that require network access are
skipped automatically when the registry or Docker is unreachable.
"""

from __future__ import annotations

import re
import shutil
import string
import subprocess
from pathlib import Path

import pytest
import requests

import brats.constants as constants
from brats.constants import (
    ALGORITHM_META_FILE_BY_ENUM,
    DUMMY_PARAMETERS,
    META_DIR,
    ZENODO_RECORD_BASE_URL,
)
from brats.utils.algorithm_config import load_algorithms, parameters_file_path

_NETWORK_ERROR_MARKERS = (
    "no such host",
    "dial tcp",
    "connection refused",
    "tls handshake timeout",
    "context deadline exceeded",
    "i/o timeout",
    "network is unreachable",
    "temporary failure in name resolution",
    "request canceled",
    "toomanyrequests",
    "too many requests",
)

# Registry responses that indicate the manifest does not exist at all.
_NOT_FOUND_ERROR_MARKERS = (
    "manifest unknown",
    "no such manifest",
    "name unknown",
    "not found",
)

# Registry responses that require authentication. These are reported but not
# treated as failures: the image may exist but be private, which cannot be
# distinguished without credentials.
_AUTH_ERROR_MARKERS = (
    "unauthorized",
    "authentication required",
    "requested access to the resource is denied",
)


def _iter_registry():
    """Yield ``(enum_cls, meta_path, algorithm_key, algorithm_data)`` tuples."""
    for enum_cls, meta_path in ALGORITHM_META_FILE_BY_ENUM.items():
        for algorithm_key, algorithm_data in load_algorithms(meta_path).items():
            yield enum_cls, meta_path, algorithm_key, algorithm_data


def _referenced_meta_paths() -> set[Path]:
    """All module-level ``Path`` constants that point into ``META_DIR``."""
    return {
        value
        for value in vars(constants).values()
        if isinstance(value, Path) and value.parent == META_DIR
    }


def _actual_meta_paths() -> set[Path]:
    return {
        file
        for file in META_DIR.iterdir()
        if file.is_file() and file.suffix in {".yml", ".yaml"}
    }


def test_every_enum_member_maps_to_metadata_entry():
    for enum_cls, meta_path in ALGORITHM_META_FILE_BY_ENUM.items():
        enum_values = {member.value for member in enum_cls}
        metadata_keys = set(load_algorithms(meta_path))
        assert enum_values == metadata_keys, (
            f"Enum/registry mismatch for {enum_cls.__name__} in {meta_path.name}: "
            f"missing in YAML: {sorted(enum_values - metadata_keys)}, "
            f"missing in enum: {sorted(metadata_keys - enum_values)}"
        )


def test_every_meta_file_is_referenced_by_a_constant():
    referenced = _referenced_meta_paths()
    actual = _actual_meta_paths()
    assert actual == referenced, (
        f"Unreferenced metadata files: {sorted(actual - referenced)}; "
        f"constants pointing to missing files: {sorted(referenced - actual)}"
    )


def test_enum_mapping_covers_all_meta_constants():
    assert set(ALGORITHM_META_FILE_BY_ENUM.values()) == _referenced_meta_paths()


def test_public_classes_use_registered_enums():
    import brats
    from brats.core.brats_algorithm import BraTSAlgorithm

    public_classes = [
        obj
        for obj in vars(brats).values()
        if isinstance(obj, type)
        and issubclass(obj, BraTSAlgorithm)
        and obj is not BraTSAlgorithm
    ]
    assert public_classes, "Expected public algorithm classes to be exported from brats"

    for cls in public_classes:
        assert hasattr(cls, "algorithm_enum"), (
            f"{cls.__name__} does not declare an algorithm_enum"
        )
        assert cls.algorithm_enum in ALGORITHM_META_FILE_BY_ENUM, (
            f"{cls.__name__}.algorithm_enum {cls.algorithm_enum.__name__} is not "
            "registered in ALGORITHM_META_FILE_BY_ENUM"
        )


def test_parameters_file_resolves_to_file_or_dummy():
    assert DUMMY_PARAMETERS.exists(), "The dummy parameters fallback is missing"
    for _, _, algorithm_key, algorithm_data in _iter_registry():
        if not algorithm_data.run_args.parameters_file:
            continue
        resolved = parameters_file_path(algorithm_data.run_args.docker_image)
        assert resolved.exists(), (
            f"parameters_file=True for {algorithm_key} but {resolved} does not exist"
        )


def test_input_name_schema_placeholders_are_valid():
    allowed_fields = {"id", "timepoint_suffix"}
    for _, _, algorithm_key, algorithm_data in _iter_registry():
        schema = algorithm_data.run_args.input_name_schema
        fields = {
            field_name
            for _, field_name, _, _ in string.Formatter().parse(schema)
            if field_name is not None
        }
        assert "id" in fields, (
            f"input_name_schema for {algorithm_key} must contain an {{id}} placeholder"
        )
        assert fields <= allowed_fields, (
            f"input_name_schema for {algorithm_key} uses unsupported placeholders: "
            f"{sorted(fields - allowed_fields)}"
        )

        suffix_by_timepoint = algorithm_data.run_args.suffix_by_timepoint
        assert ("timepoint_suffix" in fields) == bool(suffix_by_timepoint), (
            f"{algorithm_key}: the {{timepoint_suffix}} placeholder and "
            "suffix_by_timepoint must be used together"
        )

        # formatting must not raise for any combination the schema advertises
        if suffix_by_timepoint:
            for suffix in suffix_by_timepoint.values():
                schema.format(id=0, timepoint_suffix=suffix)
        else:
            schema.format(id=0)


def test_suffix_by_timepoint_values_are_valid():
    for _, _, algorithm_key, algorithm_data in _iter_registry():
        suffix_by_timepoint = algorithm_data.run_args.suffix_by_timepoint
        if suffix_by_timepoint is None:
            continue
        assert suffix_by_timepoint, f"{algorithm_key}: empty suffix_by_timepoint"
        for treatment, suffix in suffix_by_timepoint.items():
            assert treatment in {"pre", "post"}, (
                f"{algorithm_key}: unexpected treatment timepoint {treatment!r}"
            )
            assert isinstance(suffix, str) and suffix, (
                f"{algorithm_key}: invalid suffix for {treatment!r}"
            )
            assert "{" not in suffix and "}" not in suffix, (
                f"{algorithm_key}: suffix for {treatment!r} must not contain placeholders"
            )


def test_subject_modality_separator_is_valid():
    for _, _, algorithm_key, algorithm_data in _iter_registry():
        separator = algorithm_data.run_args.subject_modality_separator
        assert isinstance(separator, str) and separator, (
            f"{algorithm_key}: subject_modality_separator must be a non-empty string"
        )
        assert separator == separator.strip(), (
            f"{algorithm_key}: subject_modality_separator must not be padded with space"
        )
        assert "{" not in separator and "}" not in separator, (
            f"{algorithm_key}: subject_modality_separator must not contain placeholders"
        )


def _docker_cli_available() -> bool:
    return shutil.which("docker") is not None


def _looks_offline(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _NETWORK_ERROR_MARKERS)


def _looks_not_found(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _NOT_FOUND_ERROR_MARKERS)


def _looks_requires_auth(stderr: str) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in _AUTH_ERROR_MARKERS)


@pytest.mark.network
def test_docker_images_exist():
    """Check that every referenced image exists without pulling it."""
    if not _docker_cli_available():
        pytest.skip("docker CLI not available")

    images = sorted({data.run_args.docker_image for _, _, _, data in _iter_registry()})
    auth_required: list[str] = []
    for image in images:
        try:
            proc = subprocess.run(
                ["docker", "manifest", "inspect", "--verbose", image],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            pytest.skip(f"Could not invoke docker: {e}")

        if proc.returncode != 0:
            if _looks_offline(proc.stderr):
                pytest.skip(f"Offline, cannot resolve image {image}")
            if _looks_requires_auth(proc.stderr) and not _looks_not_found(proc.stderr):
                auth_required.append(image)
                continue
            pytest.fail(f"Container image not available: {image}\n{proc.stderr}")

        digests = re.findall(r'"digest"\s*:\s*"(sha256:[0-9a-f]+)"', proc.stdout)
        digest = digests[0] if digests else "<unknown>"
        print(f"[docker manifest] {image} -> {digest}")

    if auth_required:
        print(
            "[docker manifest] Could not verify (private or auth required): "
            + ", ".join(auth_required)
        )


@pytest.mark.network
def test_zenodo_additional_files_records_exist():
    """Check that every referenced Zenodo record exists."""
    record_ids = sorted(
        {
            data.additional_files.record_id
            for _, _, _, data in _iter_registry()
            if data.additional_files is not None
        }
    )
    assert record_ids, "Expected at least one algorithm with additional files"

    for record_id in record_ids:
        try:
            response = requests.get(f"{ZENODO_RECORD_BASE_URL}/{record_id}", timeout=30)
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Zenodo unreachable: {e}")

        if response.status_code in {502, 503, 504}:
            pytest.skip(f"Zenodo unavailable (HTTP {response.status_code})")
        assert response.status_code == 200, (
            f"Zenodo record {record_id} is not reachable (HTTP {response.status_code})"
        )
