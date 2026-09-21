# ADR-0002: YAML-Driven Algorithm Registry

**Date:** 2026-08-08
**Status:** Accepted
**Deciders**: Marcel Rosier

## Context

The package provides algorithm variants across multiple challenge tracks and BraTS
competition years. Each variant has descriptive metadata, runtime settings, and may
reference additional model files. Keeping all of this information in Python would
mix changing challenge data with the orchestration implementation.

The public API also needs stable, discoverable identifiers so users can select an
algorithm with the appropriate challenge-specific enum. The registry therefore has
to support both human-maintained metadata and a typed runtime representation.

## Decision

Store algorithm metadata in one YAML file per challenge track. Deserialize each file
into the typed dataclass model at runtime. Keep the public algorithm identifiers in
Python enums and require every enum value to have a matching metadata key.

Use YAML anchors and aliases for repeated challenge-wide defaults. Treat the YAML
files and enums as a synchronized registry rather than as independent sources.

## Rationale

- YAML keeps nested challenge metadata readable and approachable for non-developers.
- Anchors reduce repetition when many algorithms share the same defaults.
- Dataclass deserialization gives the runtime a structured representation and catches
  many missing or incompatible fields when metadata is loaded.
- Separating metadata from orchestration code reduces implementation changes when
  algorithm details change without changing the input workflow.

## Alternatives Considered

- **Hardcoded Python dataclasses:** Rejected because domain metadata would be mixed
  with implementation code and every change would require Python edits.
- **TOML or JSON:** Rejected because YAML provides better readability for this nested
  data and supports anchors for shared values.

## Consequences

- Benefits: Metadata is centralized, duplication is reduced, and existing challenge
  workflows can add algorithms without changing runner implementation.
- Drawbacks: YAML syntax and anchors introduce a learning curve, and deserialization
  adds a runtime dependency.
- Drawbacks: Enum values, metadata keys, documentation tables, and released package
  contents must remain synchronized.
- Required follow-up actions: Validate registry integrity in tests, update the enum
  and metadata together, and publish a package release for registry changes.
