[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Contributing to BraTS

First off, thanks for taking the time to contribute! 🎉


## Contribute Code
Fork the repository, clone it and implement your contribution.

**Setup:**
- We use [poetry](https://python-poetry.org/), make sure it is installed: `pip install poetry`
- Install dependencies by running: `poetry install --with dev`
- Install pre-commit hooks: `poetry run pre-commit install`
- (First time only) Run hooks against all files to catch existing issues: `poetry run pre-commit run --all-files`

**Requirements:**
- Our project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks will auto-format and lint on commit.
- Please add _meaningful_ docstring for your functions and annotate types
- Please add _meaningful_ tests for your contribution in `/tests` and make sure _all_ tests are passing by running `poetry run pytest`



Once done, create a Pull Request to integrate the code into our project!
