# AGENTS.md

## What is this package?

The BraTS orchestrator provides a typed Python API for running top-performing brain tumor segmentation and synthesis algorithms from the BraTS challenge series. Algorithms run inside Docker or Singularity containers — the package standardizes input MRI data, launches the appropriate container, and collects the output NIfTI files. Users interact with task-specific classes (e.g., `AdultGliomaPreTreatmentSegmenter`, `Inpainter`).

## Commands

```bash
uv sync                                    # install package and development dependencies
uv sync --group docs                       # additionally install documentation dependencies
uv run pytest                              # run test suite
uv run pytest --cov=brats                  # run with coverage
uv run ruff check .                        # lint
uv run ruff format --check .               # check formatting
uv run pre-commit run --all-files          # full pre-commit checks
uv run mkdocs build --strict               # validate the documentation site
```

The package supports Python 3.9+. The optional preprocessing integration requires
Python 3.10 or newer. Tests mock container execution; running the test suite does
not require a Docker daemon or a GPU.

## Architecture

The package follows a **Template Method** pattern centered on `BraTSAlgorithm` (ABC). The inference workflow is:

```
user NIfTI files → standardize inputs → run container (Docker/Singularity) → collect output
```

### Inheritance hierarchy

```
BraTSAlgorithm (ABC)                  # defines _infer_single / _infer_batch template
├── SegmentationAlgorithm (abstract)  # implements common input standardization
│   ├── SegmentationAlgorithmWith4Modalities (concrete)
│   │   └── 7 four-modality segmenters: AdultGliomaPreTreatment, Africa, GoAT, etc.
│   └── MeningiomaRTSegmenter         # T1C-only variant, overrides _standardize_batch_inputs
├── Inpainter                         # handles t1n-voided + mask inputs
└── MissingMRI                        # handles 3-of-4 modalities, synthesizes the 4th
```

There are eight concrete segmenters in total: seven four-modality segmenters and
the T1C-only `MeningiomaRTSegmenter`.

### Backend dispatch

Backend selection uses a Strategy pattern via dictionary dispatch in `_get_backend_runner()`:

- **Docker** (`brats/core/docker.py`): Default backend. For algorithms from 2024 and earlier, it uses the MLCube `/mlcube_io0` through `/mlcube_io3` mounts and runs `infer`. For 2025 and newer algorithms, it mounts `/input` and `/output` and uses the image's default command. Images are pulled from Docker Hub when needed.
- **Singularity** (`brats/core/singularity.py`): HPC-friendly backend. Converts Docker images to a sandbox, uses `--bind` for volume mounts, `--nv` for GPU support, and a temporary `--overlay` for writable storage. It follows the same year-dependent MLCube/native container split as Docker.

Both backends expose a `run_container()` function with the same caller signature
and both honor `cuda_devices`: Docker requests the given device IDs (split on
commas), while Singularity sets `SINGULARITYENV_CUDA_VISIBLE_DEVICES` around the
container run to restrict the GPUs exposed by `--nv`. IDs refer to host GPUs.

### Algorithm configuration (data-driven registry)

Algorithm metadata lives in `.yml` files under `brats/data/meta/` (one per challenge track). At runtime, YAML is deserialized via `dacite` into the `AlgorithmList` and `AlgorithmData` dataclass hierarchy (`MetaData`, `RunArgs`, `AdditionalFilesData`). The YAML files use anchors/aliases to deduplicate shared defaults. The metadata keys must match the public algorithm enum values in `brats/constants.py`.

Model weights and other additional files are downloaded from Zenodo on first use and cached locally under `brats/data/additional_files/`.

### Key supporting modules

| Module | Role |
|--------|------|
| `brats/constants.py` | `Algorithms` enums, file path constants, output schemas |
| `brats/utils/algorithm_config.py` | YAML → `AlgorithmData` deserialization |
| `brats/utils/data_handling.py` | `InferenceSetup` context manager, `input_sanity_check` |
| `brats/utils/zenodo.py` | Zenodo download and caching of model weights |
| `brats/utils/logging.py` | Singleton console handler, `enable()`/`disable()` |
| `brats/preprocessing.py` | Optional wrappers around `brainles_preprocessing` |

Inference expects preprocessed images. `input_sanity_check` verifies image shape and
logs warnings, but it does not perform registration, skull stripping, or defacing.

## Source of truth

- Public task classes are implemented in `brats/core/` and re-exported from `brats/__init__.py`.
- Public algorithm identifiers are enum members in `brats/constants.py`.
- Runtime algorithm metadata is stored in the ten `.yml` files under `brats/data/meta/`.
- Parameter files are stored under `brats/data/parameters/`. If `parameters_file` is enabled and no algorithm-specific file exists, the runner mounts the dummy parameter file.
- Model weights and other additional files are downloaded from Zenodo on first use and cached under `brats/data/additional_files/`.
- Algorithm tables in `docs/snippets/algorithm-tables/` are maintained documentation and are not generated from the YAML metadata.
- Preprocessing wrappers are implemented in `brats/preprocessing.py` and delegate to the optional `brainles_preprocessing` package.

## Adding algorithms

**To an existing challenge track:**

1. Add the algorithm identifier to the appropriate enum in `brats/constants.py`.
2. Add the matching metadata entry to the appropriate `.yml` file in `brats/data/meta/`.
3. Publish the referenced container image and add any required parameter file under `brats/data/parameters/`.
4. Update the corresponding table in `docs/snippets/algorithm-tables/`.
5. Add or update tests for configuration integrity and run the validation commands above.
6. Publish a new package release so installed users receive the enum and metadata changes.

For an existing challenge with the existing input layout, no runner or algorithm
class changes should be necessary. The YAML registry is data-driven, but it is not
independent of the public enum API.

**For a new challenge type with a novel input layout:** (1) add a `.yml` metadata file and a metadata path constant; (2) add an `Algorithms` enum subclass in `constants.py`; (3) add a concrete class inheriting from `BraTSAlgorithm` or `SegmentationAlgorithm` implementing the required input standardization and public inference methods; (4) update `Task` or preprocessing dispatch if the workflow requires it; (5) export the class in `brats/__init__.py`; (6) add tests and documentation.

## Conventions

- **Docstrings**: Google-style, parsed by `mkdocstrings` for the MkDocs site
- **Type annotations**: Required on all public methods
- **Tests**: Place tests under `tests/` following the source layout where appropriate; cross-cutting checks may live at the top level. Use `unittest.mock` for Docker/GPU mocking.
- **Exceptions**: Custom exceptions in `brats/utils/exceptions.py`
- **Linting/formatting**: `ruff` with pre-commit hooks (line length 88)

## References

- **[docs/glossary.md](docs/glossary.md)** — Domain terminology (MRI modalities, challenge types, container jargon)
- **[docs/adr/](docs/adr/)** — Architecture Decision Records
- **[mkdocs.yml](mkdocs.yml)** — Documentation navigation and build configuration
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Full contributor setup guide
