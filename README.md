# BraTS

[![Python Versions](https://img.shields.io/pypi/pyversions/brats)](https://pypi.org/project/brats/)
[![Stable Version](https://img.shields.io/pypi/v/brats?label=stable)](https://pypi.python.org/pypi/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest)](http://brats.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/brats/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/brats/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/BraTS/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/BraTS)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Providing the top performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy to use Python API powered by docker.

## Features

- Access to top-performing algorithms from recent BraTS challenges
- Easy-to-use minimal API
- Extensive documentation and examples

## Installation

With a Python 3.8+ environment, you can install `brats` directly from [PyPI](https://pypi.org/project/brats/):

```bash
pip install brats
```

> [!IMPORTANT]  
> To run `brats` you require a Docker installation. <br>
> Many algorithms also require GPU support (NVIDIA Docker). <br>
> In case you do not have access to a Cuda-capable GPU, the overview tables in the [Available Algorithms and Usage](#available-algorithms-and-usage) section indicate which algorithms are CPU compatible.




### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)


## Available Algorithms and Usage

### Segmentation
<details>

<summary> <strong> Adult Glioma Segmentation (Pre Treatment) </strong> </summary>
<br>


```python
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms

segmenter = AdultGliomaPreTreatmentSegmenter(algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm of 2023 will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.AdultGliomaPreTreatmentSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AdultGliomaPreTreatmentSegmenter))

| Year | Rank | Author                            | Paper                                      | CPU Support | Key Enum                                                                                                                             |
| ---- | ---- | --------------------------------- | ------------------------------------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 2023 | 1st  | _André Ferreira, et al._          | [Link](https://arxiv.org/abs/2402.17317v1) | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_1) |
| 2023 | 2nd  | _Andriy Myronenko, et al._        | N/A                                        | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_2) |
| 2023 | 3rd  | _Fadillah Adamsyah Maani, et al._ | [Link](https://doi.org/10.1007/978-3-031-76163-8_24) | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_3) |

</details>

<details>
<summary> <strong> Adult Glioma Segmentation Post Treatment </strong> </summary>
<br>


```python
from brats import AdultGliomaPostTreatmentSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms

segmenter = AdultGliomaPostTreatmentSegmenter(algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm of 2024 will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.AdultGliomaPostTreatmentSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AdultGliomaPostTreatmentSegmenter))

| Year | Rank | Author                   | Paper | CPU Support | Key Enum                                                                                                                          |
| ---- | ---- | ------------------------ | ----- | ----------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _André Ferreira, et al._ | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Team kimbab_            | N/A   | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Adrian Celaya_          | N/A   | &#x2705;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_3) |

</details>

<details>
<summary> <strong> BraTS-Africa Segmentation </strong> </summary>
<br>

```python
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

segmenter = AfricaSegmenter(algorithm=AfricaAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.AfricaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AfricaSegmenter))

| Year | Rank | Author                     | Paper | CPU Support | Key Enum                                                                                                        |
| ---- | ---- | -------------------------- | ----- | ----------- | --------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _Zhifan Jiang et al._      | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Long Bai, et al._         | N/A   | &#x2705;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Sarim Hashmi, et al._     | N/A   | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Andriy Myronenko, et al._ | TODO  | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Alyssa R Amod, et al._    | [Link](https://doi.org/10.1007/978-3-031-76163-8_22) | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Ziyan Huang, et al._      | [Link](https://doi.org/10.1007/978-3-031-76163-8_13) | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Meningioma Segmentation </strong> </summary>
<br>

**Note**  
Unlike other segmentation challenges the expected inputs for the Meningioma Segmentation Algorithms differ between years. 
- _2023_: All 4 modalities are used (t1c, t1n, t2f, t2w)
- _2024_: Only t1c is used  

Therefore the usage differs slightly, depending on which algorithm is used. To understand why, please refer to the [2024 challenge manuscript](https://arxiv.org/abs/2405.18383).

```python
from brats import MeningiomaSegmenter
from brats.constants import MeningiomaAlgorithms

### Example for 2023 algorithms
segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation_23.nii.gz",
)

### Example for 2024 algorithms
segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS24_1, cuda_devices="0")
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="segmentation_24.nii.gz",
)
```

**Class:** `brats.MeningiomaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.MeningiomaSegmenter))

| Year | Rank | Author                     | Paper | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | -------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _Valeria Abramova_         | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Mehdi Astaraki_           | N/A   | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Andre Ferreira, et al._   | N/A   | &#x2705;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Ziyan Huang, et al._      | [Link](https://doi.org/10.1007/978-3-031-76163-8_13) | &#x2705;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Daniel Capell'an-Mart'in et al._ | [Link](https://api.semanticscholar.org/CorpusID:272599903) | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Brain Metastases Segmentation </strong> </summary>
<br>

```python
from brats import MetastasesSegmenter
from brats.constants import MetastasesAlgorithms

segmenter = MetastasesSegmenter(algorithm=MetastasesAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.MetastasesSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.MetastasesSegmenter))

| Year | Rank | Author                     | Paper | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | -------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2023 | 1st  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Siwei Yang, et al._       | [Link](https://doi.org/10.1007/978-3-031-76163-8_17) | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Ziyan Huang, et al._      | [Link](https://doi.org/10.1007/978-3-031-76163-8_13) | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Pediatric Segmentation </strong> </summary>
<br>

```python
from brats import PediatricSegmenter
from brats.constants import PediatricAlgorithms

segmenter = PediatricSegmenter(algorithm=PediatricAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.PediatricSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.PediatricSegmenter))

| Year | Rank | Author                     | Paper | CPU Support | Key Enum                                                                                                           |
| ---- | ---- | -------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| 2024 | 1st  | _Tim Mulvany, et al._      | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Mehdi Astaraki_           | N/A   | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Sarim Hashmi, et al._     | N/A   | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Zhifan Jiang et al._      | [Link](https://api.semanticscholar.org/CorpusID:272599903) | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Andriy Myronenko, et al._ | N/A   | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Yubo Zhou_                | [Link](https://doi.org/10.1007/978-3-031-76163-8_5) | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_3) |

</details>

<details>
<summary> <strong> Generalizability Across Tumors (BraTS-GoAT) Segmentation </strong> </summary>
<br>

```python
from brats import GoATSegmenter
from brats.constants import GoATAlgorithms

segmenter = GoATSegmenter(algorithm=GoATAlgorithms.BraTS24_1, cuda_devices="0")
# these parameters are optional, by default the winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.PediatricSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.PediatricSegmenter))

| Year | Rank | Author                     | Paper | CPU Support | Key Enum                                                                                                      |
| ---- | ---- | -------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _Frank Miao, Shengjie Niu_ | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS24_1) |

</details>

### Inpainting

<details>
<summary> <strong> Inpainting </strong> </summary>
<br>

```python
from brats import Inpainter
from brats.constants import InpaintingAlgorithms

inpainter = Inpainter(algorithm=InpaintingAlgorithms.BraTS24_1, cuda_devices="0")
inpainter.infer_single(
    t1n="path/to/voided_t1n.nii.gz",
    mask="path/to/mask.nii.gz",
    output_file="inpainting.nii.gz",
)
```

**Class:** `brats.Inpainter` ([Docs](https://brats.readthedocs.io/en/latest/core/inpainting_algorithms.html#brats.core.inpainting_algorithms.Inpainter))

| Year | Rank | Author                             | Paper | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | ---------------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _Ke Chen, Juexin Zhang, Ying Weng_ | N/A   | &#x2705;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _André Ferreira, et al._           | N/A   | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Team SMINT_                       | N/A   | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Juexin Zhang, et al._             | [Link](https://doi.org/10.1007/978-3-031-76163-8_21)   | &#x2705;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Alicia Durrer, et al._            | [Link](https://doi.org/10.48550/arXiv.2402.17307)   | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Jiayu Huo, et al._                | [Link](https://doi.org/10.1007/978-3-031-76163-8_1) | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_3) |

</details>

### Missing MRI

<details>
<summary> <strong> Missing MRI </strong> </summary>
<br>

```python
from brats import MissingMRI
from brats.constants import MissingMRIAlgorithms

missing_mri = MissingMRI(algorithm=MissingMRIAlgorithms.BraTS24_1, cuda_devices="0")
# Example to synthesize t2f modality (whichever modality is missing will be inferred)
missing_mri.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    # t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="inferred_t2f.nii.gz",
)
```

**Class:** `brats.MissingMRI` ([Docs](https://brats.readthedocs.io/en/latest/core/missing_mri_algorithms.html#brats.core.missing_mri_algorithms.MissingMRI))

| Year | Rank | Author                                    | Paper | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | ----------------------------------------- | ----- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2024 | 1st  | _Jihoon Cho, Seunghyuck Park, Jinah Park_ | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Haowen Pang_                             | N/A   | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Minjoo Lim, Bogyeong Kang_               | N/A   | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_3) |

</details>

--- 

> [!TIP]
> For a full notebook example with more details please check here:  
> [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/BraTS/tutorial.ipynb)

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
