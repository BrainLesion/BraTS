# Copilot Instructions

The BraTS orchestrator is a Python package that wraps BraTS challenge brain tumor algorithms in Docker/Singularity containers. See `AGENTS.md` for the full architecture overview, commands, and conventions.

Key points:

- Package is a container orchestrator, not an ML framework — algorithms run inside Docker/Singularity, not natively
- Algorithm metadata is data-driven (YAML + dacite), not hardcoded
- Supports Python 3.9+ (avoid `X | Y` union syntax and `X | None` optional syntax)
- Public API is defined in `brats/__init__.py`
- Tests mirror source layout 1:1 under `tests/`
