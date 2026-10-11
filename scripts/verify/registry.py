"""Container registry preflight and image handling for the verification suite."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

DockerRunner = Callable[[list[str]], subprocess.CompletedProcess]

_NETWORK_ERROR_MARKERS = (
    "no such host",
    "dial tcp",
    "connection refused",
    "tls handshake timeout",
    "context deadline exceeded",
    "i/o timeout",
    "network is unreachable",
    "temporary failure in name resolution",
    "request canceled",
    "toomanyrequests",
    "too many requests",
)
_NOT_FOUND_ERROR_MARKERS = (
    "manifest unknown",
    "no such manifest",
    "name unknown",
    "not found",
)
_AUTH_ERROR_MARKERS = (
    "unauthorized",
    "authentication required",
    "requested access to the resource is denied",
)


@dataclass
class PreflightResult:
    """Outcome of a (non-pulling) registry reachability check."""

    image: str
    state: str
    """One of ``pullable``, ``not_found``, ``private`` or ``offline``."""
    size_gb: Optional[float] = None
    message: str = ""

    @property
    def pullable(self) -> bool:
        return self.state == "pullable"


def _default_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=180, check=False
    )


def _looks_like(stderr: str, markers: tuple[str, ...]) -> bool:
    lowered = stderr.lower()
    return any(marker in lowered for marker in markers)


def _manifest_layer_bytes(manifest: object) -> int:
    """Sum the layer sizes in a single image manifest, if present."""
    if not isinstance(manifest, dict):
        return 0
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        return 0
    return sum(layer.get("size", 0) for layer in layers if isinstance(layer, dict))


def _platform_is_amd64(platform: object) -> bool:
    if not isinstance(platform, dict):
        return True
    return platform.get("architecture") == "amd64" and platform.get("os") == "linux"


def _entry_manifest(entry: dict) -> object:
    for key in ("OCIManifest", "SchemaV2Manifest", "manifest"):
        if isinstance(entry.get(key), dict):
            return entry[key]
    if "layers" in entry:
        return entry
    return None


def _layer_sizes_gb(payload: object) -> Optional[float]:
    """Best-effort image size estimate from ``docker manifest inspect`` JSON.

    Only ``linux/amd64`` images are considered, matching the target server, and
    the maximum single-manifest layer total is returned.
    """
    totals: list[int] = []

    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            descriptor = entry.get("Descriptor")
            platform = (
                descriptor.get("platform") if isinstance(descriptor, dict) else None
            )
            if not _platform_is_amd64(platform):
                continue
            total = _manifest_layer_bytes(_entry_manifest(entry))
            if total:
                totals.append(total)
    elif isinstance(payload, dict):
        total = _manifest_layer_bytes(_entry_manifest(payload))
        if total:
            totals.append(total)

    if not totals:
        return None
    return max(totals) / (1000**3)


def manifest_inspect(
    image: str, *, runner: Optional[DockerRunner] = None
) -> PreflightResult:
    """Check image reachability without pulling it.

    Args:
        image (str): Container image reference
        runner (Optional[DockerRunner]): Injectable docker runner (for tests)

    Returns:
        PreflightResult: Classification and best-effort size estimate
    """
    docker_runner = runner or _default_runner
    if runner is None and shutil.which("docker") is None:
        return PreflightResult(image, "offline", message="docker CLI not found")

    try:
        proc = docker_runner(["docker", "manifest", "inspect", "--verbose", image])
    except FileNotFoundError:
        return PreflightResult(image, "offline", message="docker CLI not found")
    except (OSError, subprocess.SubprocessError) as e:
        return PreflightResult(image, "offline", message=str(e))

    if proc.returncode != 0:
        stderr = proc.stderr or ""
        if _looks_like(stderr, _NETWORK_ERROR_MARKERS):
            return PreflightResult(image, "offline", message=stderr.strip())
        if _looks_like(stderr, _AUTH_ERROR_MARKERS) and not _looks_like(
            stderr, _NOT_FOUND_ERROR_MARKERS
        ):
            return PreflightResult(image, "private", message=stderr.strip())
        if _looks_like(stderr, _NOT_FOUND_ERROR_MARKERS):
            return PreflightResult(image, "not_found", message=stderr.strip())
        return PreflightResult(image, "not_found", message=stderr.strip())

    size_gb: Optional[float] = None
    try:
        size_gb = _layer_sizes_gb(json.loads(proc.stdout))
    except (ValueError, TypeError):
        size_gb = None
    return PreflightResult(image, "pullable", size_gb=size_gb)


def image_present(image: str, *, runner: Optional[DockerRunner] = None) -> bool:
    """Return True if the image is present in the local docker daemon."""
    docker_runner = runner or _default_runner
    proc = docker_runner(["docker", "image", "inspect", image])
    return proc.returncode == 0


def resolve_image_id(
    image: str, *, runner: Optional[DockerRunner] = None
) -> Optional[str]:
    """Return the local image id for ``image``, if present."""
    docker_runner = runner or _default_runner
    proc = docker_runner(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    if proc.returncode != 0:
        return None
    image_id = (proc.stdout or "").strip()
    return image_id or None


def pull_image(
    image: str,
    *,
    runner: Optional[DockerRunner] = None,
    timeout: float = 3600.0,
) -> None:
    """Pull an image, raising ``RuntimeError`` on failure.

    Pulls can be large (several GB) and slow, so ``timeout`` defaults to one
    hour rather than the short cap used for metadata queries.
    """
    if runner is not None:
        proc = runner(["docker", "pull", image])
    else:
        proc = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"docker pull failed: {(proc.stderr or '').strip()}")
