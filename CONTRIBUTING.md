[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Contributing to BraTS

First off, thanks for taking the time to contribute!


## Contribute Code
Fork the repository, clone it and implement your contribution.

**Setup:**
- We use [uv](https://docs.astral.sh/uv/), install it via `pip install uv` or `brew install uv`
- Install dependencies by running: `uv sync`
- Install pre-commit hooks: `uv run pre-commit install`
- (First time only) Run hooks against all files to catch existing issues: `uv run pre-commit run --all-files`

**Requirements:**
- Our project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit hooks will auto-format and lint on commit.
- Please add _meaningful_ docstring for your functions and annotate types
- Please add _meaningful_ tests for your contribution in `/tests` and make sure _all_ tests are passing by running `uv run pytest`



Once done, create a Pull Request to integrate the code into our project!

## Project Documentation

- **[AGENTS.md](AGENTS.md)** — Build commands, architecture overview, source-of-truth map, and conventions (for human contributors and AI coding assistants)
- **[Architecture Decision Records](docs/adr/)** — Records of significant design decisions and their rationale
- **[Glossary](docs/glossary.md)** — Domain terminology reference (MRI modalities, challenge types, container jargon)
