# Installation

## pip install

With a Python 3.9+ environment, you can install BraTS Orchestrator directly from [PyPI](https://pypi.org/project/brats/):

```bash
# lightweight base package
pip install brats

# with preprocessing functionalities
pip install brats[preprocessing]
```

!!! important
    To run BraTS Orchestrator, you require a **Docker** installation.
    Many algorithms also require GPU support (NVIDIA Docker).
    The [algorithm tables](../reference/algorithms.md) indicate which algorithms are CPU compatible.

## Docker Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)

## Singularity Setup

[Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) is supported as an alternative to Docker. Install it following the [official guide](https://docs.sylabs.io/guides/3.0/user-guide/installation.html).

Specify the Singularity backend when running inference:

```python
from brats.constants import Backends

segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="path/to/segmentation.nii.gz",
    backend=Backends.SINGULARITY,
)
```
