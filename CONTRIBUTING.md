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

## Testing

The test suite is layered (see [ADR-0003](docs/adr/0003-layered-testing-strategy.md)):

- **Unit tests** mock all Docker/Singularity and GPU interaction and run on every
  pull request via `uv run pytest`.
- **Config-contract tests** (`tests/config/`) check that the public enums, YAML
  metadata, parameter files and name schemas stay consistent. They need no
  container daemon or GPU. Checks that require the network (container image and
  Zenodo record existence) are marked `network`; they are excluded from the
  default run and can be executed explicitly with
  `uv run pytest tests/config -m network`. They skip automatically when offline.
- **End-to-end verification** (`tests/integration/`) is opt-in and runs
  containerized algorithms on de-identified data, validating the output
  structurally. `-m integration` runs every registered algorithm; `-m smoke`
  runs a curated fast subset against the same `--data` root. Both are
  excluded from the default run through `addopts` and require a container
  backend, network access and (for most algorithms) a GPU. The suite must not be
  run under `pytest-xdist`: its GPU-free gate assumes single-process execution.

See [`tests/integration/README.md`](tests/integration/README.md) for the full
runbook (markers, options, safety model, server and background execution).

## Project Documentation

- **[AGENTS.md](AGENTS.md)** — Build commands, architecture overview, source-of-truth map, and conventions (for human contributors and AI coding assistants)
- **[Architecture Decision Records](docs/adr/)** — Records of significant design decisions and their rationale
- **[Glossary](docs/glossary.md)** — Domain terminology reference (MRI modalities, challenge types, container jargon)
