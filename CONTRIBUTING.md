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
- **End-to-end smoke tests** (`tests/integration/test_smoke.py`) are opt-in and
  execute a representative algorithm per task category on de-identified sample
  data. They are excluded from the default run through `addopts` and require a
  container backend, network access and (for most algorithms) a GPU.

### Running the smoke tests

Run the CLI directly on a GPU host (recommended):

```bash
uv run python scripts/smoke_test.py                 # all cases in the manifest
uv run python scripts/smoke_test.py --case inpainting --keep
uv run python scripts/smoke_test.py --junit-xml smoke-report.xml
uv run python scripts/smoke_test.py --force-cpu      # CPU-compatible algorithms only
```

Or through pytest (thin wrapper around the same script):

```bash
uv run pytest -m smoke
uv run pytest -m smoke -k inpainting
```

Common options and environment variables:

| Option / variable | Purpose |
|-------------------|---------|
| `--manifest PATH` | Alternative selection manifest |
| `--case ID` | Run only the given case (repeatable) |
| `--data PATH` / `BRATS_SMOKE_DATA` | Root for the cached sample data |
| `--backend docker\|singularity` / `BRATS_SMOKE_BACKEND` | Container backend |
| `--force-cpu` / `BRATS_SMOKE_FORCE_CPU` | Force CPU execution |
| `--cuda-devices` / `BRATS_SMOKE_CUDA_DEVICES` | CUDA devices to expose |
| `--timeout SECONDS` / `BRATS_SMOKE_TIMEOUT` | Per-case timeout (default: manifest `timeout`, or no limit) |
| `--keep` / `BRATS_SMOKE_KEEP` | Keep outputs and logs for debugging |
| `--junit-xml PATH` | Write a JUnit XML report |

Sample data is downloaded once from a pinned commit of
[BrainLesion/tutorials](https://github.com/BrainLesion/tutorials) and cached. The
case selection lives in [`scripts/smoke_manifest.yaml`](scripts/smoke_manifest.yaml).

The `.github/workflows/integration.yml` workflow runs the script manually
(`workflow_dispatch`) on a self-hosted GPU runner. The Python script remains the
single source of truth, so the same checks can be run directly on any server.

## Project Documentation

- **[AGENTS.md](AGENTS.md)** — Build commands, architecture overview, source-of-truth map, and conventions (for human contributors and AI coding assistants)
- **[Architecture Decision Records](docs/adr/)** — Records of significant design decisions and their rationale
- **[Glossary](docs/glossary.md)** — Domain terminology reference (MRI modalities, challenge types, container jargon)
