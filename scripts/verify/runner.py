"""Per-case orchestration for the all-algorithms verification suite."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

import brats
from brats.constants import Backends
from scripts._shared import ValidationError, time_limit
from scripts.verify import datasets
from scripts.verify.cleanup import (
    existing_weight_folders,
    new_weight_folders,
    remove_image,
    remove_path,
)
from scripts.verify.gpu import GpuError, ensure_free, parse_device_ids
from scripts.verify.models import AlgorithmCase, CaseResult, Status
from scripts.verify.registry import (
    PreflightResult,
    image_present,
    pull_image,
    resolve_image_id,
)
from scripts.verify.validation import validate_case_output
from scripts.verify.weights import weights_cache_override

DockerRunner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass
class RunOptions:
    """Options controlling a verification sweep."""

    data_root: Path
    reports_dir: Path
    workspace: Path
    cuda_devices: str = "0"
    force_cpu: bool = False
    timeout: Optional[float] = 3600.0
    pull_timeout: float = 3600.0
    min_free_gb: float = 90.0
    disk_path: Optional[Path] = None
    gpu_idle_mem_mb: int = 500
    gpu_wait_seconds: int = 600
    gpu_poll_seconds: int = 20
    keep_images: bool = False
    keep_weights: bool = False
    keep: bool = False
    fresh_weights: bool = True
    preflight: dict[str, PreflightResult] = field(default_factory=dict)
    owned_images: dict[str, Optional[str]] = field(default_factory=dict)
    docker_runner: Optional[DockerRunner] = None
    gpu_ensure: Callable[..., bool] = ensure_free


def _safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def _run_docker(
    args: list[str], runner: Optional[DockerRunner]
) -> subprocess.CompletedProcess:
    if runner is not None:
        return runner(args)
    return subprocess.run(
        args, capture_output=True, text=True, timeout=180, check=False
    )


def _docker_root_dir(runner: Optional[DockerRunner]) -> Optional[Path]:
    """Return the docker data root (image storage filesystem) if known."""
    try:
        proc = _run_docker(["docker", "info", "--format", "{{.DockerRootDir}}"], runner)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    value = (proc.stdout or "").strip()
    return Path(value) if value else None


CONTAINER_LABEL_KEY = "brats.container_label"


def _labeled_container_ids(label: str, runner: Optional[DockerRunner]) -> set[str]:
    """Return the ids of containers carrying our per-run label."""
    try:
        proc = _run_docker(
            ["docker", "ps", "-aq", "--filter", f"label={CONTAINER_LABEL_KEY}={label}"],
            runner,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    return set((proc.stdout or "").split())


def _kill_labeled_containers(label: str, runner: Optional[DockerRunner]) -> None:
    """Force-remove only the containers that carry our per-run label.

    Scoping by a unique label (rather than the image reference) guarantees we
    never touch a container started by another user, even if it uses the same
    image.
    """
    new = _labeled_container_ids(label, runner)
    if not new:
        return
    with suppress(OSError, subprocess.SubprocessError):
        _run_docker(["docker", "rm", "-f", *new], runner)


def _free_gb(paths: list[Path]) -> float:
    """Return the minimum free space (GB) across the nearest existing ancestors."""
    frees: list[float] = []
    for path in paths:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            frees.append(shutil.disk_usage(probe).free / (1000**3))
        except OSError:
            continue
    return min(frees) if frees else float("inf")


def _required_gb(options: RunOptions, image: str) -> float:
    estimate = 0.0
    preflight = options.preflight.get(image)
    if preflight is not None and preflight.size_gb is not None:
        estimate = preflight.size_gb
    return max(options.min_free_gb, estimate * 1.5 + 10.0)


def _resolve_class(class_name: str) -> type:
    cls = getattr(brats, class_name, None)
    if cls is None:
        raise RuntimeError(f"Unknown algorithm class exported from brats: {class_name}")
    return cls


def _teardown(
    case: AlgorithmCase,
    options: RunOptions,
    *,
    console: Optional[Console],
    output_dir: Optional[Path],
    weights_dir: Optional[Path],
    weights_before: Optional[set[Path]],
    remove_image_flag: bool,
    image_id: Optional[str],
) -> None:
    if not options.keep_weights:
        if weights_dir is not None:
            size = remove_path(weights_dir)
            if console:
                console.print(
                    f"[dim]removed weights cache {weights_dir} "
                    f"({size / 1e6:.1f} MB)[/dim]"
                )
        elif case.record_id and weights_before is not None:
            for folder in new_weight_folders(case.record_id, weights_before):
                size = remove_path(folder)
                if console:
                    console.print(
                        f"[dim]removed weights {folder} ({size / 1e6:.1f} MB)[/dim]"
                    )
    if remove_image_flag and not options.keep_images and image_id:
        if remove_image(image_id, runner=options.docker_runner):
            options.owned_images.pop(case.docker_image, None)
            if console:
                console.print(f"[dim]removed image {image_id}[/dim]")
        elif console:
            console.print(f"[yellow]failed to remove image {image_id}[/yellow]")
    if output_dir is not None and not options.keep:
        remove_path(output_dir)


def run_case(
    case: AlgorithmCase,
    options: RunOptions,
    *,
    console: Optional[Console] = None,
    keep_image: bool = False,
) -> CaseResult:
    """Run a single verification case end to end.

    Args:
        case (AlgorithmCase): The case to verify
        options (RunOptions): Sweep options
        console (Optional[Console]): Optional Rich console for progress output
        keep_image (bool): Keep this image even if we pulled it (a later case
            in the sweep still needs it)

    Returns:
        CaseResult: The outcome of the case
    """
    start = time.monotonic()

    def finish(
        status: Status,
        message: str = "",
        *,
        output_file: Optional[Path] = None,
        log_file: Optional[Path] = None,
        image_pulled_by_us: bool = False,
        image_id: Optional[str] = None,
    ) -> CaseResult:
        return CaseResult(
            case=case,
            status=status,
            duration_s=time.monotonic() - start,
            message=message,
            output_file=output_file,
            log_file=log_file,
            image_pulled_by_us=image_pulled_by_us,
            image_id=image_id,
        )

    if case.known_failure_reason:
        return finish(Status.EXPECTED_FAIL, case.known_failure_reason)

    try:
        algorithm_class = _resolve_class(case.class_name)
        instance = algorithm_class(
            algorithm=case.algorithm,
            cuda_devices=options.cuda_devices,
            force_cpu=options.force_cpu,
            **case.kwargs,
        )
    except Exception as e:  # noqa: BLE001 (report instantiation failures per case)
        return finish(Status.RUN_FAIL, f"{type(e).__name__}: {e}")

    run_args = getattr(instance.algorithm, "run_args", None)
    if options.force_cpu and run_args is not None and not run_args.cpu_compatible:
        return finish(
            Status.RUN_FAIL,
            f"{case.member} is not CPU-compatible; drop --force-cpu or exclude "
            "this case",
        )

    try:
        inputs = datasets.resolve_inputs(case, options.data_root)
    except datasets.DatasetError as e:
        return finish(Status.RUN_FAIL, str(e))
    missing = datasets.missing_inputs(inputs)
    if missing:
        return finish(
            Status.RUN_FAIL, f"Missing input file(s): {[str(p) for p in missing]}"
        )
    reference = datasets.resolve_reference(case, inputs)

    disk_paths = [options.data_root]
    if options.disk_path is not None:
        disk_paths.append(options.disk_path)
    # The workspace receives weights and intermediate outputs, so its filesystem
    # must have headroom too (it is often /tmp or a separate scratch volume).
    disk_paths.append(options.workspace)
    docker_root = _docker_root_dir(options.docker_runner)
    if docker_root is not None:
        disk_paths.append(docker_root)
    required = _required_gb(options, case.docker_image)
    free = _free_gb(disk_paths)
    if free < required:
        return finish(
            Status.DISK_SKIPPED,
            f"only {free:.1f} GB free (need {required:.1f} GB)",
        )

    reference_image = case.docker_image
    already_owned = reference_image in options.owned_images
    if already_owned:
        image_pulled_by_us = False
        image_id = options.owned_images[reference_image]
    else:
        image_pulled_by_us = not image_present(
            reference_image, runner=options.docker_runner
        )
        if image_pulled_by_us:
            try:
                pull_image(
                    reference_image,
                    runner=options.docker_runner,
                    timeout=options.pull_timeout,
                )
            except Exception as e:  # noqa: BLE001
                partial_id = resolve_image_id(
                    reference_image, runner=options.docker_runner
                )
                _teardown(
                    case,
                    options,
                    console=console,
                    output_dir=None,
                    weights_dir=None,
                    weights_before=None,
                    remove_image_flag=True,
                    image_id=partial_id,
                )
                return finish(
                    Status.PULL_FAIL,
                    f"{type(e).__name__}: {e}",
                    image_pulled_by_us=True,
                    image_id=partial_id,
                )
        image_id = resolve_image_id(reference_image, runner=options.docker_runner)
        if image_pulled_by_us:
            options.owned_images[reference_image] = image_id

    remove_image_flag = (
        (image_pulled_by_us or already_owned)
        and not options.keep_images
        and not keep_image
    )

    output_dir = options.workspace / _safe(case.case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.nii.gz"
    log_file = (
        options.reports_dir
        / "logs"
        / _safe(case.track)
        / (f"{_safe(case.case_id)}.log")
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not options.force_cpu:
        try:
            free_gpu = options.gpu_ensure(
                parse_device_ids(options.cuda_devices),
                idle_mem_mb=options.gpu_idle_mem_mb,
                wait_s=options.gpu_wait_seconds,
                poll_s=options.gpu_poll_seconds,
            )
        except GpuError as e:
            _teardown(
                case,
                options,
                console=console,
                output_dir=output_dir,
                weights_dir=None,
                weights_before=None,
                remove_image_flag=remove_image_flag,
                image_id=image_id,
            )
            return finish(
                Status.ENV_ERROR,
                str(e),
                output_file=output_file,
                image_pulled_by_us=image_pulled_by_us,
                image_id=image_id,
            )
        if not free_gpu:
            _teardown(
                case,
                options,
                console=console,
                output_dir=output_dir,
                weights_dir=None,
                weights_before=None,
                remove_image_flag=remove_image_flag,
                image_id=image_id,
            )
            return finish(
                Status.GPU_BUSY_SKIPPED,
                f"GPU {options.cuda_devices} still busy after "
                f"{options.gpu_wait_seconds}s",
                output_file=output_file,
                image_pulled_by_us=image_pulled_by_us,
                image_id=image_id,
            )

    weights_dir: Optional[Path] = None
    weights_before: Optional[set[Path]] = None
    if case.record_id:
        if options.fresh_weights:
            weights_dir = options.workspace / _safe(case.case_id) / "additional_files"
        else:
            weights_before = existing_weight_folders(case.record_id)

    weights_ctx = (
        weights_cache_override(weights_dir)
        if weights_dir is not None
        else nullcontext()
    )

    label = f"brats-verify-{case.slug}-{uuid.uuid4().hex}"
    previous_label = os.environ.get("BRATS_CONTAINER_LABEL")
    os.environ["BRATS_CONTAINER_LABEL"] = label
    try:
        try:
            with weights_ctx, time_limit(options.timeout):
                instance.infer_single(
                    output_file=output_file,
                    log_file=log_file,
                    backend=Backends.DOCKER,
                    **inputs,
                )
        except TimeoutError as e:
            _teardown(
                case,
                options,
                console=console,
                output_dir=output_dir,
                weights_dir=weights_dir,
                weights_before=weights_before,
                remove_image_flag=remove_image_flag,
                image_id=image_id,
            )
            return finish(
                Status.TIMEOUT,
                str(e),
                output_file=output_file,
                log_file=log_file,
                image_pulled_by_us=image_pulled_by_us,
                image_id=image_id,
            )
        except Exception as e:  # noqa: BLE001 (report any failure per case)
            _teardown(
                case,
                options,
                console=console,
                output_dir=output_dir,
                weights_dir=weights_dir,
                weights_before=weights_before,
                remove_image_flag=remove_image_flag,
                image_id=image_id,
            )
            return finish(
                Status.RUN_FAIL,
                f"{type(e).__name__}: {e}",
                output_file=output_file,
                log_file=log_file,
                image_pulled_by_us=image_pulled_by_us,
                image_id=image_id,
            )
    finally:
        if previous_label is None:
            os.environ.pop("BRATS_CONTAINER_LABEL", None)
        else:
            os.environ["BRATS_CONTAINER_LABEL"] = previous_label
        _kill_labeled_containers(label, options.docker_runner)

    try:
        validate_case_output(case, output_file, reference)
    except ValidationError as e:
        _teardown(
            case,
            options,
            console=console,
            output_dir=output_dir,
            weights_dir=weights_dir,
            weights_before=weights_before,
            remove_image_flag=remove_image_flag,
            image_id=image_id,
        )
        return finish(
            Status.INVALID_OUTPUT,
            str(e),
            output_file=output_file,
            log_file=log_file,
            image_pulled_by_us=image_pulled_by_us,
            image_id=image_id,
        )

    _teardown(
        case,
        options,
        console=console,
        output_dir=output_dir,
        weights_dir=weights_dir,
        weights_before=weights_before,
        remove_image_flag=remove_image_flag,
        image_id=image_id,
    )
    return finish(
        Status.OK,
        output_file=output_file,
        log_file=log_file,
        image_pulled_by_us=image_pulled_by_us,
        image_id=image_id,
    )


def cleanup_owned_images(
    options: RunOptions, *, console: Optional[Console] = None
) -> None:
    """Remove images we pulled that survived the sweep (safety net)."""
    if options.keep_images:
        return
    for image in list(options.owned_images):
        image_id = options.owned_images.get(image) or resolve_image_id(
            image, runner=options.docker_runner
        )
        if (
            image_id
            and remove_image(image_id, runner=options.docker_runner)
            and console is not None
        ):
            console.print(f"[dim]removed leftover image {image_id}[/dim]")
        options.owned_images.pop(image, None)
