import re
import sys
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _project_metadata() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _python_classifier_versions(classifiers: list[str]) -> set[Version]:
    versions = set()
    for classifier in classifiers:
        match = re.fullmatch(
            r"Programming Language :: Python :: (\d+\.\d+)", classifier
        )
        if match:
            versions.add(Version(match.group(1)))
    return versions


def test_python_classifiers_match_requires_python() -> None:
    project = _project_metadata()
    requirement = SpecifierSet(project["requires-python"])
    versions = _python_classifier_versions(project["classifiers"])

    assert versions
    assert all(version in requirement for version in versions)


def test_running_python_version_has_a_classifier() -> None:
    project = _project_metadata()
    versions = _python_classifier_versions(project["classifiers"])
    current_version = Version(f"{sys.version_info.major}.{sys.version_info.minor}")

    assert current_version in SpecifierSet(project["requires-python"])
    assert current_version in versions
