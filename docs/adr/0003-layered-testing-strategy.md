# ADR-0003: Layered Testing Strategy for Containerized Algorithms

**Date:** 2026-09-28
**Status:** Accepted
**Deciders**: Marcel Rosier

## Context

The core value of this package is the end-to-end inference workflow: input
standardization, backend dispatch, container execution and output collection.
The regular test suite mocks Docker/Singularity entirely, so regressions in
container images, registry metadata or the wiring between them are only caught
manually or in production. At the same time, running the real containers needs a
GPU, large downloads and network access, which makes them unsuitable for every
pull request.

We need to separate cheap checks that guard the registry contract from
resource-heavy checks that exercise real containers, without duplicating
orchestration logic between a CI workflow and a server-side script.

## Decision

Adopt a three-layer testing strategy:

1. **Unit tests** (existing): mock all container/GPU interactions and run on
   every pull request.
2. **Config-contract tests**: GPU-less and container-less, run on every pull
   request. Validate that the public enums, YAML metadata, parameter files and
   name schemas are mutually consistent. Checks that need the network (image
   existence via `docker manifest inspect`, Zenodo record existence) are marked
   `network`, excluded from the default run, and executed explicitly via
   `pytest -m network` (they skip automatically when offline).
3. **End-to-end verification tests**: opt-in via `pytest -m integration` (all
   algorithms) or `-m smoke` (a curated, sample-runnable subset). They run a
   selection of algorithms on de-identified data and validate the output
   structurally. They are excluded from default runs through `addopts`.

Keep the orchestration in a pytest-native suite under `tests/integration/`, with
the reusable pieces (case discovery, dataset mapping, GPU gate, exact-artifact
cleanup) in `scripts/verify/`. One harness runs every algorithm, is selectable
with pytest's `-k`/`-m`, and reports through pytest's terminal summary plus a
JSON artifact. There is no separate CLI to keep in sync. Wire the `smoke` subset
into CI via a `workflow_dispatch` job on a self-hosted GPU runner.

## Rationale

- Separating contract checks from end-to-end checks keeps PR CI fast and
  deterministic while still catching the most common registry regressions.
- Making the script authoritative avoids drift between a CI-only code path and
  what maintainers run by hand.
- Excluding network checks from the default run keeps PR CI fast and
  deterministic; a single dedicated CI job runs them explicitly, and they skip
  gracefully when the external services are unreachable.
- An opt-in marker means contributors are never blocked by missing GPUs, while
  maintainers can still gate releases on a full smoke run.

## Alternatives Considered

- **Run smoke tests on every PR:** Rejected because GPU, network and large image
  downloads make this slow, flaky and unavailable on standard runners.
- **Put orchestration only in a CI workflow:** Rejected because it cannot be run
  directly on a server and duplicates logic that the suite already needs.
- **A bespoke sweep CLI:** Rejected in favour of a pytest-native harness; pytest
  already provides case selection, markers, result aggregation and reporting,
  so a separate CLI/result model only added drift-prone duplication.
- **Golden-baseline regression (Dice/HD95):** Deferred as a deliberate
  follow-up; structural validation is the first, cheaper step.

## Consequences

- Benefits: registry regressions surface in PR CI; a single command verifies a
  real container end to end; the suite degrades gracefully when offline.
- Drawbacks: two test modes to maintain, and the network checks depend on
  external services (skipped, not failed, when unavailable).
- Required follow-up actions: keep the curated `smoke` subset and the dataset
  mapping in `scripts/verify/` representative, update the ADR and CONTRIBUTING
  when the server or runner specifics change, and add golden-baseline
  regression separately if needed.
