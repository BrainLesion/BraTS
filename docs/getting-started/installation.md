# Installation

## pip install

With a Python 3.9+ environment, you can install BraTS Orchestrator directly from [PyPI](https://pypi.org/project/brats/):

```bash
# via pip
pip install brats
pip install brats[preprocessing]

# or via uv
uv add brats
uv add 'brats[preprocessing]'
```

!!! important
    BraTS Orchestrator requires **Docker** or **Singularity** to run algorithm containers.
    Most algorithms also need GPU support (NVIDIA Container Toolkit).
    The [algorithm tables](../reference/algorithms.md) indicate which algorithms are CPU compatible.

## Docker Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)

## Singularity Setup

[Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) provides a container runtime alternative to Docker, commonly used on HPC clusters. Install it following the [official guide](https://docs.sylabs.io/guides/3.0/user-guide/installation.html).

Specify the Singularity backend when running inference:

```python
from brats.constants import Backends

segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="path/to/segmentation.nii.gz",
    backend=Backends.SINGULARITY,
)
```
