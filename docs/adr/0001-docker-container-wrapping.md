# Container orchestration via Template Method

## Status

Accepted

## Context and Problem Statement

BraTS challenge organizers and participants provide algorithms as container images (Docker/MLCube format). These containers come from diverse research teams and vary in their CLI conventions, volume mount layouts, GPU requirements, and supported modalities. The package must provide a single, typed Python API that works uniformly across all algorithms without modifying or re-implementing the containers themselves.

Adding a new algorithm should be possible without writing Python code — only a metadata entry and a published container image should be required (for existing challenge types). New challenge types with novel input layouts should require minimal Python additions.

## Considered Options

- **Option A: Reimplement algorithms natively** — Extract model code and weights from each container, port them into a shared Python/ML environment, and run inference natively.

- **Option B: Fork and normalize containers** — Modify each container image to conform to a standardized CLI and volume layout before publishing.

- **Option C: Wrap containers as-is with an orchestration layer** — Keep containers untouched. Build a Python orchestration layer that standardizes inputs before invoking the container, dispatches to Docker or Singularity, and collects/renames outputs.

## Decision Outcome

**Chose Option C: Wrap containers with orchestration via Template Method and Strategy patterns.**

Rationale:

- **Zero container modification**: Containers are used exactly as published by challenge participants. No forking, no re-verification of algorithm correctness.
- **Template Method for workflow**: `BraTSAlgorithm` (ABC) defines the inference skeleton — standardize inputs, run container, collect output. Subclasses (`SegmentationAlgorithm`, `Inpainter`, `MissingMRI`) implement challenge-specific input standardization. This keeps the orchestration logic centralized while allowing per-challenge customization.
- **Strategy for backends**: A dispatch dictionary in `_get_backend_runner()` selects Docker or Singularity at runtime. Both backends accept the same `run_container()` signature, making the choice transparent to the algorithm classes.
- **YAML-only additions**: For existing challenge types, adding an algorithm requires only a YAML entry in `brats/data/meta/` and a published container image — no Python code changes.

Option A was rejected because maintaining native environments for 60+ heterogeneous algorithms would be unsustainable and would require re-validating every algorithm. Option B was rejected because modifying containers breaks reproducibility and shifts maintenance burden to this package.

## Consequences

**Positive:**

- Algorithms run in their original environment — no re-validation needed
- Adding a new algorithm to an existing challenge type is a YAML-only change
- Backend choice (Docker vs Singularity) is transparent to algorithm classes
- Input standardization absorbs container-to-container differences in naming conventions

**Negative:**

- Users must have Docker installed and running (or Singularity for HPC)
- Container image pulls on first use add latency
- The Docker and Singularity backends share helper functions that currently live in `docker.py`, creating tight coupling between the two backend modules (tracked in [#162](https://github.com/BrainLesion/BraTS/issues/162))
- Year-to-year changes in container layout (MLCube → native mounts) require branching logic in both backends
- Debugging requires inspecting container logs rather than native Python stack traces
