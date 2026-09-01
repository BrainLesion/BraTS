# Backends

BraTS Orchestrator supports two backends for running algorithm containers.

## Docker (Default)

Docker is the default backend. No special code configuration is needed — just a working Docker installation.

```python
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import Backends

segmenter = AdultGliomaPreAndPostTreatmentSegmenter(cuda_devices="0")
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
    backend=Backends.DOCKER,  # default
)
```

### GPU Acceleration

Most algorithms require GPU acceleration. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed, then pass `cuda_devices` to specify which GPU to use:

```python
segmenter = AdultGliomaPreAndPostTreatmentSegmenter(cuda_devices="0")
```

Multiple GPUs can be selected with a comma-separated string, e.g. `cuda_devices="0,1"`.

## Singularity

[Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) is fully supported for environments where Docker isn't available (e.g., HPC clusters).

```python
from brats.constants import Backends

segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
    backend=Backends.SINGULARITY,
)
```

### GPU Acceleration

Singularity uses the `--nv` flag to expose host GPUs — no additional toolkit is needed. The `cuda_devices` parameter is honored: the orchestrator sets `SINGULARITYENV_CUDA_VISIBLE_DEVICES` around the container run, so only the requested GPUs are visible inside it:

```python
segmenter = AdultGliomaPreAndPostTreatmentSegmenter(cuda_devices="0,1")
```

Notes:

- `cuda_devices` refers to **host** GPU IDs. If the host already defines `CUDA_VISIBLE_DEVICES` (e.g., set by a batch scheduler such as SLURM), the `cuda_devices` value takes precedence inside the container.
- Manually setting `SINGULARITYENV_CUDA_VISIBLE_DEVICES` is no longer necessary; the `cuda_devices` parameter overrides it.

---

Check the [algorithm tables](../reference/algorithms.md) to see which algorithms support CPU-only execution.
