# Backends

BraTS Orchestrator supports two backends for running algorithm containers.

## Docker (Default)

Docker is the default backend. No special configuration is needed beyond a working Docker installation.

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
    # backend=Backends.DOCKER is the default, so this line is optional:
    backend=Backends.DOCKER,
)
```

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

## GPU Support

Most algorithms require GPU acceleration. Pass `cuda_devices` to specify which GPU to use:

```python
segmenter = AdultGliomaPreAndPostTreatmentSegmenter(cuda_devices="0")
```

!!! warning "Singularity GPU selection"
    The Singularity backend uses the `--nv` flag, which [exposes all host GPUs](https://docs.sylabs.io/guides/latest/user-guide/gpu.html#multiple-gpus) regardless of `cuda_devices`. To limit GPUs with Singularity, set `SINGULARITYENV_CUDA_VISIBLE_DEVICES` before running. See [issue #164](https://github.com/BrainLesion/BraTS/issues/164).

Check the [algorithm tables](../reference/algorithms.md) to see which algorithms support CPU-only execution.
