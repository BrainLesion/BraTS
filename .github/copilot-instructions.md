# Copilot Instructions

The BraTS orchestrator is a Python package that wraps BraTS challenge brain tumor algorithms in Docker/Singularity containers. Read `AGENTS.md` for the full architecture overview, commands, source-of-truth map, and contribution workflow.

Key points:

- Package is a container orchestrator, not an ML framework — algorithms run inside Docker/Singularity, not natively
- Algorithm metadata is data-driven (`brats/data/meta/*.yml` + dacite), while selectable algorithm identifiers remain public enums in `brats/constants.py`
- Supports Python 3.9+ (avoid `X | Y` union syntax and `X | None` optional syntax)
- Public task classes are implemented in `brats/core/` and re-exported from `brats/__init__.py`
- For existing tracks, adding an algorithm requires updating both the enum and matching metadata, documentation tables, tests, and the released package
- Container mounts and commands differ between algorithms from 2024 and earlier and algorithms from 2025 onward
- Tests are mostly organized under `tests/` by source area and mock container/GPU execution
- Documentation uses MkDocs and `mkdocstrings`; validate changes with `uv run mkdocs build --strict`
