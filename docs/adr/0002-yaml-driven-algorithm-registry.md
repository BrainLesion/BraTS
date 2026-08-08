# YAML-driven algorithm registry

## Status

Accepted

## Context and Problem Statement

The package must provide access to 60+ algorithm variants across 10 challenge tracks (segmentation, inpainting, missing MRI), spanning multiple BraTS competition years (2023–2025). Each algorithm has associated metadata (authors, paper links, challenge rank), runtime arguments (Docker image, input name schema, shared memory requirements, CPU compatibility), and optional additional files (model weights hosted on Zenodo).

Hardcoding this information in Python would be verbose, error-prone, and would require a code change and a new release for every algorithm addition.

## Considered Options

- **Option A: Hardcoded Python dataclasses** — Define algorithm configurations as Python dataclass instances in `.py` files. Add new algorithms by writing Python code.

- **Option B: YAML files with dacite deserialization** — Store algorithm configurations as YAML files in `brats/data/meta/`. At runtime, parse YAML into typed dataclasses using the `dacite` library. One YAML file per challenge track.

- **Option C: TOML or JSON configuration** — Use TOML (Python's `tomllib`) or JSON for algorithm metadata.

## Decision Outcome

**Chose Option B: YAML files with dacite deserialization.**

Rationale:

- **Human-readable and writable**: YAML is more readable than JSON for structured nested data and supports comments.
- **YAML anchors/aliases**: Challenge-wide defaults (input name schema, challenge name, rank indicators) can be deduplicated using YAML `&anchors` and `*aliases`, avoiding repetition across algorithm entries in the same file.
- **Type-safe deserialization**: `dacite.from_dict()` maps the parsed YAML dict directly to the `AlgorithmList` → `AlgorithmData` → `MetaData`/`RunArgs`/`AdditionalFilesData` dataclass hierarchy. Schema mismatches are caught at load time.
- **Zero-code algorithm additions**: Adding a new algorithm to an existing challenge requires only a YAML entry — no Python changes, no new release needed if the Docker image is already published.

Option A was rejected because it would embed domain data in source code, making it harder for non-developers to contribute algorithms. Option C (TOML) was considered but YAML was preferred for its anchor/alias support and better readability for deep nesting. The `dacite` dependency adds a small cost but provides schema validation that plain `yaml.safe_load()` alone would not.

## Consequences

**Positive:**

- Adding a new algorithm to an existing challenge track is a YAML-only change
- YAML anchors reduce duplication within each challenge's metadata file
- `dacite` catches schema errors at load time rather than at container execution time
- Non-developers (e.g., challenge organizers or algorithm authors) can contribute algorithm metadata

**Negative:**

- `dacite` is an additional runtime dependency
- YAML anchor/alias syntax has a learning curve for contributors unfamiliar with it
- Runtime deserialization means schema errors in YAML files are only caught when the package is used, not at build time
- The Python code still needs separate `Algorithms` enum subclasses and (for new challenge types with unique input layouts) new segmenter classes — the YAML layer does not fully eliminate code changes
