import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
DOCS_INDEX = REPOSITORY_ROOT / "docs" / "index.md"
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
PYTHON_BADGE = "https://img.shields.io/pypi/pyversions/brats"
PYTHON_BADGE_MARKDOWN = (
    f"[![Python Versions]({PYTHON_BADGE})](https://pypi.org/project/brats/)"
)


class TestPythonSupportMetadata(unittest.TestCase):
    def test_python_badge_uses_dynamic_pypi_metadata_in_readme_and_docs(self):
        self.assertIn(PYTHON_BADGE_MARKDOWN, README.read_text(encoding="utf-8"))
        self.assertIn(PYTHON_BADGE_MARKDOWN, DOCS_INDEX.read_text(encoding="utf-8"))

    def test_pyproject_declares_supported_python_versions(self):
        pyproject_text = PYPROJECT.read_text(encoding="utf-8")

        self.assertIn('requires-python = ">=3.9,<4.0"', pyproject_text)

        for version in ("3.9", "3.10", "3.11", "3.12", "3.13", "3.14"):
            self.assertIn(
                f'"Programming Language :: Python :: {version}"', pyproject_text
            )
