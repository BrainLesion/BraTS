"""End-to-end orchestration tests for ``run_case`` with mocked docker/GPU."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.verify import runner as runner_module
from scripts.verify.cases import build_cases
from scripts.verify.models import Status
from scripts.verify.runner import RunOptions, run_case


class _FakeAlgorithm:
    def __init__(self, algorithm, cuda_devices="0", force_cpu=False, **kwargs):
        self.algorithm = algorithm

    def infer_single(self, output_file, log_file=None, backend=None, **inputs):
        reference = next(iter(inputs.values()))
        img = nib.load(str(reference))
        data = np.ones(img.shape, dtype=np.int16)
        nib.save(nib.Nifti1Image(data, img.affine), str(output_file))
        if log_file is not None:
            Path(log_file).write_text("fake log")


def _fake_docker_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")


def _case(case_id: str):
    return next(c for c in build_cases() if c.case_id == case_id)


def _options(tmp_path: Path, **overrides):
    defaults = dict(
        data_root=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        workspace=tmp_path / "workspace",
        force_cpu=True,
        min_free_gb=0.0,
        docker_runner=_fake_docker_runner,
    )
    defaults.update(overrides)
    return RunOptions(**defaults)


def _materialize_inputs(case, data_root: Path) -> None:
    from scripts.verify.datasets import resolve_inputs

    for path in resolve_inputs(case, data_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), str(path))


def test_expected_fail_short_circuits(monkeypatch, tmp_path: Path):
    case = _case("inpainting.BraTS23_1")
    case = replace(case, known_failure_reason="not published")
    result = run_case(case, _options(tmp_path))
    assert result.status is Status.EXPECTED_FAIL
    assert result.message == "not published"


def test_successful_run_pulls_validates_and_cleans_up(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    assert not case.requires_weights
    _materialize_inputs(case, tmp_path / "data")

    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _FakeAlgorithm)
    monkeypatch.setattr(
        runner_module, "image_present", lambda image, runner=None: False
    )
    monkeypatch.setattr(
        runner_module,
        "pull_image",
        lambda image, runner=None, timeout=3600.0: None,
    )
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:deadbeef"
    )
    removed: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "remove_image",
        lambda image_id, runner=None: removed.append(image_id) or True,
    )

    result = run_case(case, _options(tmp_path))
    assert result.status is Status.OK, result.message
    assert result.image_pulled_by_us
    assert removed == ["sha256:deadbeef"]
    assert result.output_file is not None
    assert not result.output_file.exists()  # temp dir removed unless --keep
    assert result.log_file is not None and result.log_file.exists()


def test_run_failure_is_reported(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    _materialize_inputs(case, tmp_path / "data")

    class _Boom(_FakeAlgorithm):
        def infer_single(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _Boom)
    monkeypatch.setattr(runner_module, "image_present", lambda image, runner=None: True)
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:x"
    )

    result = run_case(case, _options(tmp_path))
    assert result.status is Status.RUN_FAIL
    assert "boom" in result.message


def test_gpu_busy_is_skipped(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    _materialize_inputs(case, tmp_path / "data")
    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _FakeAlgorithm)
    monkeypatch.setattr(runner_module, "image_present", lambda image, runner=None: True)
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:x"
    )

    result = run_case(
        case,
        _options(tmp_path, force_cpu=False, gpu_ensure=lambda ids, **kwargs: False),
    )
    assert result.status is Status.GPU_BUSY_SKIPPED


def test_missing_inputs_reported(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _FakeAlgorithm)

    result = run_case(case, _options(tmp_path))
    assert result.status is Status.RUN_FAIL
    assert "Missing input" in result.message


def test_keep_retains_output(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    _materialize_inputs(case, tmp_path / "data")
    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _FakeAlgorithm)
    monkeypatch.setattr(runner_module, "image_present", lambda image, runner=None: True)
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:x"
    )

    result = run_case(case, _options(tmp_path, keep=True))
    assert result.status is Status.OK
    assert result.output_file is not None and result.output_file.exists()


def test_kill_labeled_containers_only_removes_ours():
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess:
        calls.append(args)
        if "ps" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="c-ours c-ours2", stderr=""
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    runner_module._kill_labeled_containers("run-1", runner)
    ps_calls = [c for c in calls if "ps" in c]
    rm_calls = [c for c in calls if "rm" in c]
    assert ps_calls == [
        ["docker", "ps", "-aq", "--filter", "label=brats.container_label=run-1"]
    ]
    assert len(rm_calls) == 1
    assert rm_calls[0][:3] == ["docker", "rm", "-f"]
    assert set(rm_calls[0][3:]) == {"c-ours", "c-ours2"}


def test_kill_labeled_containers_noop_when_none():
    calls: list[list[str]] = []

    def runner(args: list[str]) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    runner_module._kill_labeled_containers("run-1", runner)
    assert [c for c in calls if "rm" in c] == []


def test_run_failure_still_kills_containers(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    _materialize_inputs(case, tmp_path / "data")

    class _Boom(_FakeAlgorithm):
        def infer_single(self, *args, **kwargs):
            raise RuntimeError("boom")

    calls: list[list[str]] = []

    def docker_runner(args: list[str]) -> subprocess.CompletedProcess:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="c-new", stderr=""
        )

    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _Boom)
    monkeypatch.setattr(runner_module, "image_present", lambda image, runner=None: True)
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:x"
    )

    result = run_case(case, _options(tmp_path, docker_runner=docker_runner))
    assert result.status is Status.RUN_FAIL
    ps_calls = [c for c in calls if "ps" in c]
    rm_calls = [c for c in calls if "rm" in c]
    assert any("label=brats.container_label=" in " ".join(c) for c in ps_calls)
    assert rm_calls and rm_calls[0][:3] == ["docker", "rm", "-f"]


def test_fresh_weights_redirected_and_cleaned(monkeypatch, tmp_path: Path):
    import brats.constants as constants
    import brats.utils.zenodo as zenodo

    case = _case("inpainting.BraTS23_1")
    assert case.record_id is not None
    _materialize_inputs(case, tmp_path / "data")

    real_cache = constants.ADDITIONAL_FILES_FOLDER
    real_before = (
        set(real_cache.glob(f"{case.record_id}_v*.*.*"))
        if real_cache.exists()
        else set()
    )
    original_override = zenodo.ADDITIONAL_FILES_FOLDER

    class _WeightAlgorithm(_FakeAlgorithm):
        def infer_single(self, output_file, log_file=None, backend=None, **inputs):
            record_dir = zenodo.ADDITIONAL_FILES_FOLDER / f"{case.record_id}_v1.0.0"
            record_dir.mkdir(parents=True, exist_ok=True)
            (record_dir / "weights.bin").write_bytes(b"weights")
            super().infer_single(
                output_file, log_file=log_file, backend=backend, **inputs
            )

    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _WeightAlgorithm)
    monkeypatch.setattr(runner_module, "image_present", lambda image, runner=None: True)
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:x"
    )

    options = _options(tmp_path)
    result = run_case(case, options)
    assert result.status is Status.OK, result.message
    # Override is restored and the real cache is untouched.
    assert original_override == zenodo.ADDITIONAL_FILES_FOLDER
    real_after = (
        set(real_cache.glob(f"{case.record_id}_v*.*.*"))
        if real_cache.exists()
        else set()
    )
    assert real_before == real_after
    # Freshly downloaded weights (in the temp cache) were removed.
    assert not list(options.workspace.rglob("weights.bin"))


def test_keep_image_defers_removal(monkeypatch, tmp_path: Path):
    case = _case("pediatric.BraTS25_1")
    _materialize_inputs(case, tmp_path / "data")
    monkeypatch.setattr(runner_module, "_resolve_class", lambda name: _FakeAlgorithm)
    monkeypatch.setattr(
        runner_module, "image_present", lambda image, runner=None: False
    )
    monkeypatch.setattr(
        runner_module,
        "pull_image",
        lambda image, runner=None, timeout=3600.0: None,
    )
    monkeypatch.setattr(
        runner_module, "resolve_image_id", lambda image, runner=None: "sha256:z"
    )
    removed: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "remove_image",
        lambda image_id, runner=None: removed.append(image_id) or True,
    )

    options = _options(tmp_path)
    first = run_case(case, options, keep_image=True)
    assert first.status is Status.OK
    assert removed == []
    assert options.owned_images.get(case.docker_image) == "sha256:z"

    second = run_case(case, options, keep_image=False)
    assert second.status is Status.OK
    assert removed == ["sha256:z"]
    assert case.docker_image not in options.owned_images
