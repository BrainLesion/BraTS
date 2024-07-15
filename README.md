[![PyPI version brats-algorithms](https://badge.fury.io/py/brats.svg)](https://pypi.python.org/pypi/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest)](http://brats.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/brats/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/brats/actions/workflows/tests.yml)

# BraTS

Providing the top performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy to use Python API powered by docker.

## Features

- Access to top-performing algorithms from recent BraTS challenges
- Easy-to-use minimal API
- Extensive documentation and examples

## Installation

With a Python 3.8+ environment, you can install `brats` directly from [PyPI](https://pypi.org/project/brats/):

```sh
pip install brats
```

### Docker and NVIDIA Container Toolkit Setup

To run `brats` you need a working Docker installation.
Most algorithms also require GPU support (NVIDIA Docker). 

Installation instructions:
- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit) 

## Use Cases and Tutorials

A minimal example to create a segmentation could look like this:

```python
from brats import AdultGliomaSegmenter
segmenter = AdultGliomaSegmenter()
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

For more examples and details please refer to our extensive Notebook tutorials [**TODO**]

## Contributing

We welcome contributions from the community, including bug reports, feature requests, and code contributions. 
For more information on how to contribute, please see our [CONTRIBUTING.md](CONTRIBUTING.md) file.
