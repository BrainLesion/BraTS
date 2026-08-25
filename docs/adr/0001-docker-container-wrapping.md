# ADR-0001: Container Orchestration via Template Method and Strategy

**Date:** 2026-08-08
**Status:** Accepted
**Deciders**: Marcel Rosier

## Context

BraTS challenge organizers and participants provide algorithms as container images.
These containers come from different research teams and vary in command-line
conventions, input layouts, GPU requirements, and supported modalities. The package
needs to provide one typed Python API without modifying or reimplementing the
algorithms themselves.

The design must also support both local Docker environments and HPC environments
where Singularity is preferred. Adding an algorithm to an existing challenge should
not require changes to the orchestration workflow when its input layout is unchanged.

## Decision

Keep participant containers unchanged and wrap them with a Python orchestration
layer. The shared inference workflow standardizes inputs, invokes the selected
container backend, and collects the result.

Use the Template Method pattern for the shared inference workflow. Task-specific
classes provide the input standardization and public inference interfaces. Use the
Strategy pattern to select Docker or Singularity at runtime behind a common backend
contract.

## Rationale

- Preserving the original containers supports reproducibility and avoids revalidating
  modified algorithm implementations.
- A shared workflow gives users a consistent API despite differences between
  challenge submissions.
- Separating task behavior from backend execution keeps challenge-specific logic out
  of the container runners.
- Supporting both backends makes the package usable on workstations and HPC systems.

## Alternatives Considered

- **Native reimplementation:** Rejected because maintaining separate model
  environments and revalidating dozens of algorithms would be unsustainable.
- **Fork and normalize containers:** Rejected because modifying submissions would
  reduce reproducibility and move maintenance into this package.

## Consequences

- Benefits: Algorithms run in their published environments, users get one API, and
  backend choice remains independent of task-specific code.
- Drawbacks: Users need a compatible container runtime, first-use image retrieval can
  be slow, and failures often require inspecting container logs.
- Drawbacks: Container conventions can change between challenge years, so both
  backends must preserve their shared contract while supporting those differences.
- Required follow-up actions: Keep the backend contract aligned and document new
  container conventions when challenge formats change.
