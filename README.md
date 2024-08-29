[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/brats?logo=python&color=2EC553)](https://pypi.org/project/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest&color=2EC553)](http://brats.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/brats/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/brats/actions/workflows/tests.yml)
[![PyPI version brats-algorithms](https://img.shields.io/pypi/v/brats?color=2EC553)](https://pypi.python.org/pypi/brats/)
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
from brats.utils.constants import AdultGliomaAlgorithms

segmenter = AdultGliomaSegmenter(algorithm=AdultGliomaAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

For more examples and details please refer to our extensive Notebook tutorials here [NBViewer](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/BraTS/tutorial.ipynb) ([GitHub](https://github.com/BrainLesion/tutorials/blob/main/BraTS/tutorial.ipynb)). For the best experience open the notebook in Colab.

## Algorithms

<details open>

<summary> <strong> Adult Glioma Segmentation </strong> </summary>
<br>

**Class:** `brats.AdultGliomaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.segmentation_algorithms.AdultGliomaSegmenter))


| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AdultGliomaAlgorithms)|
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _André Ferreira, et al._ | [Link](https://arxiv.org/abs/2402.17317v1) | &#x274C; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AdultGliomaAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Andriy Myronenko, et al._ | N/A | &#x274C; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AdultGliomaAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Fadillah Adamsyah Maani, et al._ | N/A | &#x274C; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AdultGliomaAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> BraTS-Africa Segmentation </strong> </summary>
<br>

**Class:** `brats.AfricaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.segmentation_algorithms.AfricaSegmenter))

| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AfricaAlgorithms)|
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _Andriy Myronenko, et al._ | TODO | &#x274C; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AfricaAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Alyssa R Amod, et al._ | N/A | &#x274C; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AfricaAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Ziyan Huang, et al._ | N/A | &#x2705; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.AfricaAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Meningioma Segmentation </strong> </summary>
<br>

**Class:** `brats.MeningiomaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.segmentation_algorithms.MeningiomaSegmenter))


| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MeningiomaAlgorithms)|
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _Andriy Myronenko, et al._ | N/A | &#x274C; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MeningiomaAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Ziyan Huang, et al._ | N/A | &#x2705; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MeningiomaAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Zhifan Jiang et al._ | N/A | &#x274C; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MeningiomaAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Brain Metastases Segmentation </strong> </summary>
<br>

**Class:** `brats.MetastasesSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.segmentation_algorithms.MetastasesSegmenter))

| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MetastasesAlgorithms)|
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _Andriy Myronenko, et al._ | N/A | &#x274C; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MetastasesAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Siwei Yang, et al._ | N/A | &#x274C; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MetastasesAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Ziyan Huang, et al._ | N/A | &#x2705; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.MetastasesAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Pediatric Segmentation </strong> </summary>
<br>

**Class:** `brats.PediatricSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.segmentation_algorithms.PediatricSegmenter))


| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.PediatricAlgorithms)|
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _Zhifan Jiang et al._ | N/A | &#x274C; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.PediatricAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Andriy Myronenko, et al._ | N/A | &#x274C; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.PediatricAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Yubo Zhou_ | N/A | &#x274C; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.PediatricAlgorithms.BraTS23_3) |

</details>

---

<details>
<summary> <strong> Inpainting </strong> </summary>
<br>

**Class:** `brats.Inpainter` ([Docs](https://brats.readthedocs.io/en/latest/algorithms.html#brats.core.inpainting_algorithms.Inpainter))

| Year | Rank | Author | Paper | CPU Support | [Key Enum](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.InpaintingAlgorithms) |
|------|-------|--------|-------|-------------|-------------|
| 2023 | 1st | _Juexin Zhang, et al._ | N/A | &#x2705; | [BraTS23_1](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.InpaintingAlgorithms.BraTS23_1) |
| 2023 | 2nd | _Alicia Durrer, et al._ | N/A | &#x274C; | [BraTS23_2](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.InpaintingAlgorithms.BraTS23_2) |
| 2023 | 3rd | _Jiayu Huo, et al._ | N/A | &#x2705; | [BraTS23_3](https://brats.readthedocs.io/en/latest/constants.html#brats.utils.constants.InpaintingAlgorithms.BraTS23_3) |


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
