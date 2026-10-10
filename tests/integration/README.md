# Integration verification suite

Opt-in, container/GPU-backed verification that every registered BraTS algorithm
is **pullable**, **runnable** with the package's own configuration, and produces
**structurally sensible** output. This is an integration smoke level: it does
*not* check segmentation accuracy (no Dice/golden baseline).

The suite is pytest-native — there is no separate CLI. All safety-critical logic
(GPU gate, exact-artifact cleanup, disk guard, weight isolation) lives in
[`scripts/verify/`](../../scripts/verify) and is invoked from fixtures/tests.

## Markers

| Marker | Selects |
|--------|---------|
| `integration` | Every registered algorithm case (excluded from default runs) |
| `smoke` | Curated, fast subset runnable from auto-downloaded sample data |
| `network` | Zero-disk container-image reachability preflight (no pulls) |

Default `addopts` excludes `integration` and `network`, so a plain
`uv run pytest` stays fast and GPU-free.

## Prerequisites

- A working Docker daemon and `docker login` (for images/registry access).
- `nvidia-smi` and a free GPU for most algorithms (pass `--force-cpu` only for
  CPU-compatible algorithms).
- Network access to Docker Hub, Zenodo and GitHub (sample data).
- The suite **must not run under `pytest-xdist`** (`-n`): the GPU-free gate
  assumes single-process execution and raises a usage error otherwise.

## Data

- Full run: point `--data` at a preprocessed data root (default `~/brats_data`).
- Sample run: `--download-sample-data` fetches a pinned
  [BrainLesion/tutorials](https://github.com/BrainLesion/tutorials) subject and
  runs only the `smoke` subset.

## Commands

```bash
# Zero-disk reachability preflight for every container image
uv run pytest tests/integration -m network

# Every registered algorithm against a local data root
uv run pytest tests/integration -m integration --data ~/brats_data --cuda-devices 1

# Curated, fast subset using auto-downloaded sample data
uv run pytest tests/integration -m smoke --download-sample-data

# Narrow the selection
uv run pytest tests/integration -m integration -k inpainting
```

## Options

| Option / variable | Purpose |
|-------------------|---------|
| `--data PATH` / `BRATS_VERIFY_DATA` | Root folder containing the challenge data (default `~/brats_data`) |
| `--download-sample-data` | Download the small sample dataset and run only the `smoke` subset |
| `--cuda-devices IDS` / `BRATS_VERIFY_CUDA_DEVICES` | CUDA devices to expose (default `0`) |
| `--force-cpu` | Force CPU execution (only CPU-compatible algorithms) |
| `--timeout SECONDS` | Per-case timeout (0 disables the limit) |
| `--min-free-gb GB` / `BRATS_VERIFY_MIN_FREE_GB` | Free-space floor checked per filesystem (default 30) |
| `--disk-path PATH` | Extra filesystem to include in the free-space check |
| `--workspace PATH` / `BRATS_VERIFY_WORKSPACE` | Folder for per-case outputs/weights (default: a temp dir) |
| `--use-cached-weights` | Use the shared Zenodo cache instead of fresh downloads (cache is never deleted) |
| `--gpu-idle-mem-mb` / `--gpu-wait-seconds` / `--gpu-poll-seconds` | GPU-free gate tuning |
| `--keep` | Keep the per-case workspace and logs for debugging |
| `--verify-report PATH` / `BRATS_VERIFY_REPORT` | JSON report location (default `reports/verify-results.json`) |
| `--junitxml PATH` | Write a JUnit XML report (pytest built-in) |

## Safety model

- **GPU:** before *every* case the target GPU must have no compute process,
  ≤ `--gpu-idle-mem-mb` (default 500 MB) used and 0% utilization; otherwise the
  case is skipped after `--gpu-wait-seconds`. Other users' jobs are never
  interrupted. Containers are pinned to `--cuda-devices`.
- **Cleanup:** only artifacts this run created are removed — images pulled by us
  (exact image id, never forced), freshly downloaded weight folders, and
  containers tagged with a per-run label. Pre-existing images/weights and other
  users' containers are never touched.
- **Disk:** free space is checked on the data root, `--disk-path`, the workspace
  and the Docker root before each case.
- **RAM:** weight archives stream to disk while extracting (no whole-archive
  buffering).

## Server runbook

```bash
git fetch origin
git worktree add ~/brats-verify feature/all-algorithms-verification-suite
cd ~/brats-verify && uv sync --frozen

# Keep weights/outputs off /tmp and on a large, non-tmpfs filesystem
export BRATS_VERIFY_WORKSPACE="$HOME/brats-verify-work"
mkdir -p "$BRATS_VERIFY_WORKSPACE"
df -h "$BRATS_VERIFY_WORKSPACE" /var/lib/docker

# 1) zero-disk reachability pass (no pulls)
uv run pytest tests/integration -m network -q

# 2) one guarded case, then verify cleanup
uv run pytest tests/integration -m integration \
    -k "inpainting.BraTS23_1" --cuda-devices 1 \
    --data ~/brats_data --timeout 1800 --keep -q
docker ps -a; docker images; ls brats/data/additional_files

# 3) curated sample subset, then the full sweep
uv run pytest tests/integration -m smoke --download-sample-data --cuda-devices 1 -q
uv run pytest tests/integration -m integration \
    --data ~/brats_data --cuda-devices 1 --timeout 7200
```

After each session: `df -h /`, `docker images` (no leftover test images),
`ls brats/data/additional_files` (only pre-existing cache remains).

## Running in the background

Long sweeps should survive an SSH disconnect. Use `tmux` (or `screen`):

```bash
tmux new -s brats-verify
# ... run the pytest command ...
# detach with: Ctrl-b d     reattach with: tmux attach -t brats-verify
```

Or `nohup`, which keeps running after logout:

```bash
nohup uv run pytest tests/integration -m integration \
    --data ~/brats_data --cuda-devices 1 --timeout 7200 \
    > "$BRATS_VERIFY_WORKSPACE/verify.log" 2>&1 &
disown
tail -f "$BRATS_VERIFY_WORKSPACE/verify.log"      # follow progress
```

The JSON report (`reports/verify-results.json`) is written when the session
finishes; the terminal/JSON summary is the structured result for automation.

## CI

[`.github/workflows/integration.yml`](../../.github/workflows/integration.yml)
runs the `smoke` subset manually (`workflow_dispatch`) on a self-hosted GPU
runner.
