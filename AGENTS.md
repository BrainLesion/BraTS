# AGENTS.md

## What is this package?

The BraTS orchestrator provides a typed Python API for running top-performing brain tumor segmentation and synthesis algorithms from the BraTS challenge series. Algorithms run inside Docker or Singularity containers — the package standardizes input MRI data, launches the appropriate container, and collects the output NIfTI files. Users interact with task-specific classes (e.g., `AdultGliomaPreTreatmentSegmenter`, `Inpainter`).

## Commands

```bash
uv sync                                    # install all dependencies
uv run pytest                              # run test suite
uv run pytest --cov=brats                  # run with coverage
uv run ruff check .                        # lint
uv run ruff format --check .               # check formatting
uv run pre-commit run --all-files          # full pre-commit checks
```

## Architecture

The package follows a **Template Method** pattern centered on `BraTSAlgorithm` (ABC). The inference workflow is:

```
user NIfTI files → standardize inputs → run container (Docker/Singularity) → collect output
```

### Inheritance hierarchy

```
BraTSAlgorithm (ABC)                  # defines _infer_single / _infer_batch template
├── SegmentationAlgorithm (abstract)  # implements input standardization for T1c/T1n/T2f/T2w
│   ├── SegmentationAlgorithmWith4Modalities (concrete)
│   │   └── 7 concrete segmenters: AdultGliomaPreTreatment, Africa, GoAT, etc.
│   └── MeningiomaRTSegmenter         # T1C-only variant, overrides _standardize_batch_inputs
├── Inpainter                         # handles t1n-voided + mask inputs
└── MissingMRI                        # handles 3-of-4 modalities, synthesizes the 4th
```

### Backend dispatch

Backend selection uses a Strategy pattern via dictionary dispatch in `_get_backend_runner()`:

- **Docker** (`brats/core/docker.py`): Default backend. Pulls images from Docker Hub, mounts data/weights/output volumes, runs `infer` command.
- **Singularity** (`brats/core/singularity.py`): HPC-friendly backend. Converts Docker images to sandbox format, uses `--bind` for volume mounts, `--nv` for GPU, and `--overlay` for writable storage.

Both backends expose a `run_container()` function with the same caller signature.

### Algorithm configuration (data-driven registry)

Algorithm metadata lives in YAML files under `brats/data/meta/` (one per challenge track). At runtime, YAML is deserialized via `dacite` into the `AlgorithmData` dataclass hierarchy (`MetaData`, `RunArgs`, `AdditionalFilesData`). The YAML files use anchors/aliases to deduplicate shared defaults.

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

## Adding algorithms

**To an existing challenge track:** Add an entry to the appropriate YAML file in `brats/data/meta/`. No Python changes needed — just publish the Docker image and reference it.

**For a new challenge type with a novel input layout:** (1) Add a YAML metadata file; (2) add an `Algorithms` enum subclass in `constants.py`; (3) create a concrete class inheriting from `BraTSAlgorithm` or `SegmentationAlgorithm` implementing `_standardize_single_inputs`/`_standardize_batch_inputs` and `infer_single`/`infer_batch`; (4) export the class in `brats/__init__.py`.

## Conventions

- **Docstrings**: Google-style, used by Sphinx/napoleon for ReadTheDocs
- **Type annotations**: Required on all public methods
- **Tests**: Mirror source layout 1:1 under `tests/`. Use `unittest.mock` for Docker/GPU mocking.
- **Exceptions**: Custom exceptions in `brats/utils/exceptions.py`
- **Linting/formatting**: `ruff` with pre-commit hooks (line length 88)

## References

- **[docs/glossary.md](docs/glossary.md)** — Domain terminology (MRI modalities, challenge types, container jargon)
- **[docs/adr/](docs/adr/)** — Architecture Decision Records
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Full contributor setup guide
