[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/brats?logo=python&color=2EC553)](https://pypi.org/project/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest)](http://brats.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/brats/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/brats/actions/workflows/tests.yml)
[![PyPI version brats-algorithms](https://badge.fury.io/py/brats.svg)](https://pypi.python.org/pypi/brats/)
[![License: GPLv3](https://img.shields.io/badge/License-AGPLv3-blue.svg?color=2EC553)](https://www.gnu.org/licenses/agpl-3.0)

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

## Algorithms

<details open>

<summary> <strong> Adult Glioma Segmentation </strong> </summary>
<br>

| Year | Rank | Author                            | Paper                                      | CPU Support |
| ---- | ---- | --------------------------------- | ------------------------------------------ | ----------- |
| 2023 | 1st  | _André Ferreira, et al._          | [Link](https://arxiv.org/abs/2402.17317v1) | &#x274C;    |
| 2023 | 2nd  | _Andriy Myronenko, et al._        | N/A                                        | &#x274C;    |
| 2023 | 3rd  | _Fadillah Adamsyah Maani, et al._ | N/A                                        | &#x274C;    |

</details>

<details>
<summary> <strong> BraTS-Africa Segmentation </strong> </summary>
<br>

| Year | Rank | Author                     | Paper | CPU Support |
| ---- | ---- | -------------------------- | ----- | ----------- |
| 2023 | 1st  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    |
| 2023 | 2nd  | _Alyssa R Amod, et al._    | N/A   | &#x274C;    |
| 2023 | 3rd  | _Ziyan Huang, et al._      | N/A   | &#x2705;    |

</details>

<details>
<summary> <strong> Meningioma Segmentation </strong> </summary>
<br>

| Year | Rank | Author                     | Paper | CPU Support |
| ---- | ---- | -------------------------- | ----- | ----------- |
| 2023 | 1st  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    |
| 2023 | 2nd  | _Ziyan Huang_              | N/A   | &#x2705;    |
| 2023 | 3rd  | _Zhifan Jiang et al._      | N/A   | &#x274C;    |

</details>

<details>
<summary> <strong> Brain Metastases Segmentation </strong> </summary>
<br>

| Year | Rank | Author | Paper | CPU Support |
|------|-------|--------|-------|-------------|
| 2023 | 1st | _Andriy Myronenko, et al._ | N/A | &#x274C; |
| 2023 | 2nd | _Siwei Yang, et al._ | N/A | &#x274C; |
| 2023 | 3rd | _Ziyan Huang, et al._ | N/A | &#x2705; |

</details>

<details>
<summary> <strong> Pediatric Segmentation </strong> </summary>
<br>

| Year | Rank | Author                     | Paper | CPU Support |
| ---- | ---- | -------------------------- | ----- | ----------- |
| 2023 | 1st  | _Zhifan Jiang et al._      | N/A   | &#x274C;    |
| 2023 | 2nd  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    |
| 2023 | 3rd  | _Yubo Zhou_                | N/A   | &#x274C;    |

</details>

## Citation

If you use BraTS in your research, please cite it to support the development!

```
TODO: citation will be added asap
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests and Questions

Please open a new issue [here](https://github.com/BrainLesion/BraTS/issues).

### Code contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
