# Implementation Plan: All-Algorithms Verification Suite (`scripts/verify`)

**Status:** proposed
**Base branch:** `185-feature-integration-smoke-test-suite-for-containerized-algorithms`
(PR #186, head `7119b66`)
**Feature branch:** `feature/all-algorithms-verification-suite` (stacked PR, base = the
branch above)

> This document is the handoff spec for implementing the verification suite. It is
> intentionally implementation-focused and is not part of the published docs nav.

## 0. Context

The base branch added the layered testing foundation:

- `ALGORITHM_META_FILE_BY_ENUM` in `brats/constants.py` (enum class → metadata YAML).
- `BraTSAlgorithm.algorithm_enum` `ClassVar` on every concrete class.
- `parameters_file_path()` in `brats/utils/algorithm_config.py`.
- `scripts/smoke_test.py` + `scripts/smoke_manifest.yaml` (opt-in E2E smoke runner).
- `tests/config/test_registry_contract.py`, ADR-0003, and CONTRIBUTING testing notes.

The verification suite depends on this refactor, so it is a **stacked PR**. If the base
branch has already merged into `main` before work starts, branch off `main` instead.

## 1. Goal

For **every** algorithm the package offers, verify the container is:

1. **pullable** (image resolves on the registry),
2. **runnable** with the package's own configuration on real data,
3. **structurally sensible** in its output.

The algorithm set is enumerated dynamically (currently **72 members across 10 tracks**;
**16** require Zenodo weights; years span 2023–2026). Do not hardcode counts in logic.

Requirements:

- Slice runs **by challenge track** for focused reports, plus one **overall** report.
- Persist a **JSON** report; print **Rich** console tables. **No HTML.**
- **No CI wiring.**
- **One subject per track.**
- **Delete images and weights after each run**, precisely and safely (see §6).
- **Check the target GPU is free immediately before every run** (see §7) — critical.
- **No negative tests**; do not use `broken_data/` or `oslo/`.

Out of scope: golden-baseline/Dice, batch inference, HTML, LRU image caching,
`docker image prune -a`.

## 2. Server environment (runbook facts)

- Host `a4000-21an1`, user `marcelrosier`; `uv` at `~/.local/bin/uv` (login shell only).
- Docker 23.0.1; Docker Hub `docker login` is configured.
- GPUs: `0` = RTX A5000 24 GB, `1` = Quadro RTX 8000 48 GB. Target **GPU 1**
  (`--cuda-devices 1`). Both can be claimed at any time (`ollama_medgemma`,
  `open-webui` containers are running; another user uses GPU 0).
- Data root: `~/brats_data/` (see §4).
- `/` is tight and shared (Docker root `/var/lib/docker`); use the disk guard.
- Weights cache is `brats/data/additional_files/` **inside the checkout** — run from a
  dedicated worktree so it is isolated and safely deletable.
- Run from a worktree: `git worktree add ~/brats-verify <branch>` then `uv sync --frozen`.

## 3. Architecture

```
scripts/
├─ known_failures.yaml            # image ref -> human reason
└─ verify/
   ├─ __main__.py                 # CLI entrypoint (python -m scripts.verify)
   ├─ models.py                   # dataclasses + Status enum
   ├─ discovery.py                # introspect brats -> AlgorithmCase list, grouped by track
   ├─ datasets.py                 # track -> DatasetSpec, resolve input paths
   ├─ gpu.py                      # GPU-free gate
   ├─ registry.py                 # manifest preflight, image presence/size, pull
   ├─ runner.py                   # per-case orchestration
   ├─ cleanup.py                  # exact-artifact teardown
   ├─ validation.py               # structural output checks
   └─ reporting.py                # Rich console + JSON persistence
```

Conventions: Google-style docstrings, type annotations on public functions, `ruff`
(line length 88), no comments unless needed, conventional commits. `scripts/` is linted
but excluded from the sdist. No new runtime deps (`rich`, `PyYAML` already present).

Reuse `scripts/smoke_test.py` helpers where clean (`check_environment`,
output validation, input building). Prefer extracting genuinely shared code into a small
`scripts/_shared.py` imported by both; a small refactor of `smoke_test.py` on this branch
is acceptable. Avoid copy-paste drift.

## 4. Data mapping (verified working)

Root `~/brats_data`. `infer_single` reassigns internal names, so only per-modality
**file paths** matter (`brats/core/brats_algorithm.py:254`,
`brats/core/segmentation_algorithms.py:79`).

| Track (enum class) | Dir / subject | Inputs (package kwargs) | Notes |
|---|---|---|---|
| `AdultGliomaPreTreatmentAlgorithms` | `adult_glioma/BraTS-GLI-00001-000` | `t1c,t1n,t2f,t2w` | files are unprefixed (`t1c.nii.gz`) |
| `AdultGliomaPreAndPostTreatmentAlgorithms` | `adult_glioma` (pre) / `adult_glioma_postop/BraTS-GLI-02073-100` (post) | 4-mod; `treatment_timepoint` | |
| `MeningiomaAlgorithms` | `adult_meningioma/BraTS-MEN-00000-000` | 4-mod | skip `cBraTS-MEN-00171-000` |
| `MeningiomaRTAlgorithms` | `adult_meningioma_24/BraTS-MEN-RT-0010-1` | `t1c` | filename uses `_` separator |
| `PediatricAlgorithms` | `pediatric/BraTS-PED-00030-000` | 4-mod | |
| `AfricaAlgorithms` | `africa/BraTS-SSA-00126-000` | 4-mod | |
| `MetastasesAlgorithms` | `metastases/BraTS-MET-00013-000` (≤2024) / `MET25/BraTS-MET-00833-000` (2025) | 4-mod | select by `year` |
| `GoATAlgorithms` | `adult_glioma/BraTS-GLI-00001-000` | 4-mod | generic source |
| `InpaintingAlgorithms` | `inpainting/BraTS-GLI-00001-000` | `t1n`=`...-t1n-voided.nii.gz`, `mask`=`...-mask.nii.gz` | |
| `MissingMRIAlgorithms` | `missing_mri/BraTS-GLI-00001-000` (synth `t2w`) | `t1c,t1n,t2f` | exactly 3 inputs |

All files load as valid NIfTI (mostly `(240,240,155)`; `adult_glioma_postop`
`(182,218,182)`, `adult_meningioma_24` `(512,512,321)`).

## 5. Data models (`models.py`)

```python
class Status(str, Enum):
    OK = "ok"
    EXPECTED_FAIL = "expected_fail"
    PREFLIGHT_FAIL = "preflight_fail"
    PULL_FAIL = "pull_fail"
    DISK_SKIPPED = "disk_skipped"
    GPU_BUSY_SKIPPED = "gpu_busy_skipped"
    RUN_FAIL = "run_fail"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    ENV_ERROR = "env_error"

@dataclass(frozen=True)
class AlgorithmCase:
    track: str            # enum class name, e.g. "InpaintingAlgorithms"
    member: str           # e.g. "BraTS23_1"
    class_name: str       # public class, e.g. "Inpainter"
    algorithm: Algorithms
    docker_image: str
    year: int
    task: str
    dataset_key: str
    kwargs: dict[str, object]
    requires_weights: bool
    parameters_file: bool
    known_failure_reason: str | None = None

@dataclass
class CaseResult:
    case: AlgorithmCase
    status: Status
    duration_s: float
    message: str = ""
    output_file: Path | None = None
    image_pulled_by_us: bool = False
    image_id: str | None = None

@dataclass
class RunReport:
    results: list[CaseResult]
    started: datetime
    finished: datetime
    # helpers: by_track(), by_status(), summary()
```

## 6. Execution flow

### Phase 0 — preflight (no pulls)
Authenticated `docker manifest inspect <image>` for selected cases; classify
`pullable / not_found / private`; apply `known_failures.yaml` (`expected_fail`).
`--preflight-only` persists/prints and exits (zero disk).

### Phase 1 — sweep (sequential, per case)
```
1. skip if expected_fail
2. disk guard: free_gb >= max(--min-free-gb, expected_image_size_gb * 1.5 + buffer)
3. pull image if absent (record whether we pulled it + resolve image id)
4. GPU-FREE GATE   <-- immediately before run, every case
5. infer_single(output_file, log_file, backend, **inputs, **kwargs) under timeout
6. validate output structurally
7. cleanup exact artifacts created by this case
8. record CaseResult; continue on failure
```

### Phase 2 — report
Rich per-track tables + overall summary; write JSON (and `summary.md`).

## 7. GPU-free gate (`gpu.py`) — CRITICAL

Runs **after the pull and immediately before `infer_single`**, for **every** case, since
contention can appear mid-sweep. Cheap (<100 ms).

```python
def parse_device_ids(cuda_devices: str) -> list[int]
def ensure_free(ids: list[int], *, idle_mem_mb: int = 500,
                wait_s: int = 600, poll_s: int = 20,
                sleep=time.sleep) -> bool
```

- Read `nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used
  --format=csv,noheader,nounits` and `nvidia-smi --query-compute-apps=gpu_uuid,pid,
  used_memory --format=csv,noheader,nounits`.
- Free iff: no compute process on that UUID, `memory.used <= idle_mem_mb`,
  `utilization == 0`.
- Busy → poll every `poll_s` up to `wait_s`; still busy → `GPU_BUSY_SKIPPED` (do not
  run; still clean up anything pulled). Never interrupts others.
- `--force-cpu` skips the gate (CPU-compatible algorithms only).
- `nvidia-smi` missing / invalid index → `ENV_ERROR`.

## 8. Cleanup (`cleanup.py`) — exact artifacts only

Do **not** use before/after directory snapshots (a concurrent install could be deleted).
Track identity:

- **Image:** before pulling, `docker image inspect <ref>`. If absent, pull and capture
  the resolved `.Id`; after validation `docker rmi <Id>` (exact ID). If it was already
  present, leave it untouched.
- **Weights:** the case knows its `additional_files.record_id`; resolve the exact folder
  `additional_files/<record_id>_v*.*.*`. If absent before the run, the download is ours →
  delete exactly that folder after. If present, leave it. No global snapshot.
- **Temp:** per-case temp dir we created → delete unless `--keep`.
- Log every deletion (path + size). Flags: `--keep-images`, `--keep-weights`, `--keep`.

## 9. Validation (`validation.py`)

Adapt the structural checks from `scripts/smoke_test.py::_validate_output`:

- Output exists, non-empty, readable NIfTI; finite; not all-zero.
- Shape and affine match a reference input (first input).
- Segmentation: integer labels within `allowed_labels` (default `[0,1,2,3,4]`, per-case
  override) with at least one non-background voxel.
- MissingMRI / inpainting: intensity image → finiteness + shape/affine.

## 10. Reporting (`reporting.py`)

- **Rich:** one table per track (`member | year | status | time | detail`), then an
  overall summary (`track | total | ok | expected_fail | failed | skipped`) and a
  consolidated failure detail.
- **Persistent** (`--reports-dir`, default `reports/<UTC-timestamp>/`):
  - `results.json` — all `CaseResult`s + aggregates + timestamps + versions + options.
  - `summary.md` — human-readable aggregate.
  - `logs/<track>/<member>.log` — `infer_single` log.

JSON shape (abridged):
```json
{
  "started": "...", "finished": "...", "backend": "docker", "cuda_devices": "1",
  "summary": {"total": 72, "ok": 0, "expected_fail": 4, "failed": 0, "skipped": 0},
  "tracks": {"InpaintingAlgorithms": {"total": 15, "ok": 0, "failed": 0,
                                       "skipped": 0, "expected_fail": 0}},
  "results": [
    {"track": "InpaintingAlgorithms", "member": "BraTS23_1", "class": "Inpainter",
     "year": 2023, "image": "brainles/brats23_inpainting_ying_weng:latest",
     "status": "ok", "duration_s": 33.0, "message": "", "output_file": "..."}
  ]
}
```

## 11. CLI (`__main__.py`)

```
uv run python -m scripts.verify                        # all tracks
uv run python -m scripts.verify --preflight-only        # zero-disk reachability check
uv run python -m scripts.verify --no-run                # resolve cases + inputs only
uv run python -m scripts.verify --track inpainting --track missing_mri
uv run python -m scripts.verify --year 2025 --cuda-devices 1
uv run python -m scripts.verify --data ~/brats_data --min-free-gb 90
```

Flags: `--track` (repeatable), `--year`, `--case`, `--data`, `--backend {docker,singularity}`,
`--cuda-devices`, `--force-cpu`, `--timeout`, `--min-free-gb`, `--gpu-idle-mem-mb`,
`--gpu-wait-seconds`, `--gpu-poll-seconds`, `--keep-images`, `--keep-weights`, `--keep`,
`--reports-dir`, `--preflight-only`, `--no-run`.

Defaults: `--cuda-devices "0"`, `--timeout 3600`, `--min-free-gb 90`,
`--gpu-idle-mem-mb 500`, `--gpu-wait-seconds 600`, `--gpu-poll-seconds 20`, strict delete,
`--reports-dir reports/<timestamp>`.

## 12. Known failures (`known_failures.yaml`)

Seed with the four GoAT 2025 images (verified `no such manifest`, not published):

```yaml
"brainles/brats25_goat_cmengynn:latest": "container not published (verified 404)"
"brainles/brats25_goat_cmuh_aic:latest": "container not published (verified 404)"
"brainles/brats25_goat_satushe:latest": "container not published (verified 404)"
"brainles/brats25_goat_stockholmfederation:latest": "container not published (verified 404)"
```

Reported as `expected_fail`. If one becomes pullable, flag it in the report.

## 13. Testing the tool

- Unit tests under `tests/verify/` for pure logic:
  - `discovery.build_cases()` → current counts (72 cases, 10 tracks) and stable mapping.
  - `datasets.resolve_inputs()` against a temp fixture tree.
  - `gpu.parse_device_ids()` and `ensure_free()` with injected fake `nvidia-smi` output.
  - `cleanup` exact-target selection (only ours).
- `--no-run` and `--preflight-only` must not download/pull anything.
- No CI workflows.

## 14. Server runbook (after implementation)

```bash
git fetch origin
git worktree add ~/brats-verify feature/all-algorithms-verification-suite
cd ~/brats-verify && uv sync --frozen

# 1) zero-disk reachability pass
uv run --frozen python -m scripts.verify --preflight-only --cuda-devices 1 --data ~/brats_data

# 2) focused first run (image likely already present -> quick)
uv run --frozen python -m scripts.verify --track inpainting --cuda-devices 1 --data ~/brats_data

# 3) full sweep (long; ~tens of GB per image re-downloaded, hours to days)
uv run --frozen python -m scripts.verify --cuda-devices 1 --data ~/brats_data \
    --reports-dir ~/verify-reports
```

After each session check: `df -h /`, `docker images` (no leftover test images),
`ls brats/data/additional_files` (only pre-existing cache remains).

## 15. Acceptance criteria

- `--preflight-only` classifies all images; the 4 GoAT are `expected_fail`, rest
  `pullable`.
- A single-track run produces `results.json` + Rich output; images we pulled are removed;
  weight folders we downloaded are removed; pre-existing images/weights untouched.
- GPU gate blocks/skips when the target GPU is busy and never runs concurrently.
- Validation catches malformed/empty outputs.
- `ruff check .` and `ruff format --check .` clean; default `pytest` green.

## 16. Open items / assumptions

1. `--reports-dir` default `reports/<timestamp>/` in the repo, gitignored.
2. `--min-free-gb` default 90 as a floor combined with a dynamic image-size estimate.
3. Extract shared helpers into `scripts/_shared.py` and have `smoke_test.py` import them
   (small refactor on this branch), or duplicate locally — prefer extraction.
4. Revisit `known_failures.yaml` when containers are republished.
