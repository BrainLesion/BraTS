# BraTS Orchestrator

[![Python Versions](https://img.shields.io/pypi/pyversions/brats)](https://pypi.org/project/brats/)
[![Stable Version](https://img.shields.io/pypi/v/brats?label=stable)](https://pypi.python.org/pypi/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest)](http://brats.readthedocs.io/?badge=latest)
[![tests](https://github.com/BrainLesion/brats/actions/workflows/tests.yml/badge.svg)](https://github.com/BrainLesion/brats/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BrainLesion/BraTS/graph/badge.svg?token=A7FWUKO9Y4)](https://codecov.io/gh/BrainLesion/BraTS)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

```
______         _____ _____
| ___ \       |_   _/  ___|
| |_/ /_ __ __ _| | \ `--.
| ___ \ '__/ _` | |  `--. \
| |_/ / | | (_| | | /\__/ /
\____/|_|  \__,_\_/ \____/


 _____          _               _             _
|  _  |        | |             | |           | |
| | | |_ __ ___| |__   ___  ___| |_ _ __ __ _| |_ ___  _ __
| | | | '__/ __| '_ \ / _ \/ __| __| '__/ _` | __/ _ \| '__|
\ \_/ / | | (__| | | |  __/\__ \ |_| | | (_| | || (_) | |
 \___/|_|  \___|_| |_|\___||___/\__|_|  \__,_|\__\___/|_|
```

Providing the top-performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy-to-use Python API powered by Docker.

## Features

- Access to top-performing algorithms from recent BraTS challenges
- Easy-to-use minimal API
- Extensive documentation and examples

## Installation

With a Python 3.8+ environment, you can install BraTS orchestrator directly from [PyPI](https://pypi.org/project/brats/):

```bash
# lightweight base package
pip install brats
# With preprocessing functionalities
pip install brats[preprocessing]
```

> [!IMPORTANT]  
> To run BraTS orchestrator, you require a Docker installation. <br>
> Many algorithms also require GPU support (NVIDIA Docker). <br>
> In case you do not have access to a CUDA-capable GPU, the overview tables in the [Available Algorithms and Usage](#available-algorithms-and-usage) section indicate which algorithms are CPU compatible.




### Docker and NVIDIA Container Toolkit Setup

- **Docker**: Installation instructions on the official [website](https://docs.docker.com/get-docker/)
- **NVIDIA Container Toolkit**: Refer to the [NVIDIA install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and the official [GitHub page](https://github.com/NVIDIA/nvidia-container-toolkit)

## Singularity Support
BraTS orchestrator also supports Singularity as an alternative to Docker.  
To enable Singularity, install it following the [official guide](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) and specify the backend to use as `Backends.SINGULARITY` when running the inference:
```python
from brats.constants import Backends
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="path/to/segmentation.nii.gz",
    backend=Backends.SINGULARITY
)
```

## Available Algorithms and Usage

> [!IMPORTANT]
> BraTS challenge algorithms require preprocessed brain images. See section [Data preprocessing requirements](#data-preprocessing-requirements)

### Segmentation Challenges
<img src="https://github.com/BrainLesion/brats/blob/main/figures/segmentation_fig.png?raw=true" alt="matched_instance_figure" height="250"/>

> Note: Some legacy segmentation algorithms from BraTS challenges before 2023 are available via [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit).
<br>


#### Adult Glioma Segmentation (Pre & Post-Treatment) 
> Adult Glioma Segmentation on pre- and post-treatment brain MRI exams.  
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms

segmenter = AdultGliomaPreAndPostTreatmentSegmenter(algorithm=AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```
> Note: If you're interested in Adult Glioma Segmentation, the [BrainLes GlioMODA package](https://github.com/BrainLesion/GlioMODA?tab=readme-ov-file#gliomoda) may also be of interest.
<br>

**Class:** `brats.AdultGliomaPreAndPostTreatmentSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AdultGliomaPreAndPostTreatmentSegmenter))
<br>
**Challenge Paper 2023:** [Link](https://arxiv.org/abs/2107.02314)
<br>

| Year | Rank | Author                   | Paper                                    | CPU Support | Key Enum                                                                                                                                  |
| ---- | ---- | ------------------------ | ---------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Ishika Jain, et al._    | N/A                                      | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1)   |
| 2025 | 2nd  | _Qu Lin, et al._         | N/A                                      | &#x2705;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_2)   |
| 2025 | 3rd  | _Liwei Jin, et al._      | N/A                                      | &#x2705;    | [BraTS25_3A](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_3A) |
| 2025 | 3rd  | _Adrian Celaya, et al._  | N/A                                      | &#x274C;    | [BraTS25_3B](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_3B) |
| 2024 | 1st  | _André Ferreira, et al._ | N/A                                      | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_1)         |
| 2024 | 2nd  | _Heejong Kim, et al._    | [Link](https://arxiv.org/abs/2409.08143) | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_2)         |
| 2024 | 3rd  | _Adrian Celaya_          | N/A                                      | &#x2705;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AdultGliomaPostTreatmentAlgorithms.BraTS24_3)         |


> Note: The MNI152 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in the 2024 and subsequent BraTS Glioma Post-treatment Segmentation challenges.

</details>
<br>

#### Adult Glioma Segmentation (Pre-Treatment) 
> Adult Glioma Segmentation on pre-treatment brain MRI exams.  
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms

segmenter = AdultGliomaPreTreatmentSegmenter(algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```
> Note: If you're interested in Adult Glioma Segmentation, the [BrainLes GlioMODA package](https://github.com/BrainLesion/GlioMODA?tab=readme-ov-file#gliomoda) may also be of interest.
<br>

**Class:** `brats.AdultGliomaPreTreatmentSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AdultGliomaPreTreatmentSegmenter))
<br>
**Challenge Paper 2023:** [Link](https://arxiv.org/abs/2107.02314)
<br>

| Year | Rank | Author                            | Paper                                                | CPU Support | Key Enum                                                                                                                             |
| ---- | ---- | --------------------------------- | ---------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 2023 | 1st  | _André Ferreira, et al._          | [Link](https://arxiv.org/abs/2402.17317v1)           | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_1) |
| 2023 | 2nd  | _Andriy Myronenko, et al._        | [Link](https://arxiv.org/abs/2510.25058)             | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_2) |
| 2023 | 3rd  | _Fadillah Adamsyah Maani, et al._ | [Link](https://doi.org/10.1007/978-3-031-76163-8_24) | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.**AdultGliomaPreTreatmentAlgorithms**.BraTS23_3) |

</details>

> Note: The SRI24 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in the 2023 and prior BraTS Glioma Pre-Treatment Segmentation challenges.
<br>


#### BraTS-Africa Segmentation
> Adult Glioma Segmentation on brain MRI exams in Sub-Sahara-Africa patient population.  
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

segmenter = AfricaSegmenter(algorithm=AfricaAlgorithms.BraTS25_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.AfricaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.AfricaSegmenter))
<br>
**Challenge Paper 2023** [Link](https://doi.org/10.48550/arXiv.2305.19369)
<br>
**Challenge Paper 2024**: N/A
<br>

| Year | Rank | Author                          | Paper                                                | CPU Support | Key Enum                                                                                                          |
| ---- | ---- | ------------------------------- | ---------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Claudia Takyi Ankomah, et al._ | N/A                                                  | &#x2705;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS25_1)   |
| 2025 | 2nd  | _William Boonzaier, et al._     | N/A                                                  | &#x274C;    | [BraTS25_2A](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS25_2A) |
| 2025 | 2nd  | _Mohtady Barakat, et al._       | N/A                                                  | &#x274C;    | [BraTS25_2B](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS25_2B) |
| 2025 | 3rd  | _Ahmed Jaheen, et al._          | N/A                                                  | &#x274C;    | [BraTS25_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS25_3)   |
| 2024 | 1st  | _Abhijeet Parida, et al._       | [Link](https://arxiv.org/abs/2412.04111)             | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_1)   |
| 2024 | 2nd  | _Yanguang Zhao, et al._         | [Link](https://doi.org/10.48550/arXiv.2410.18698)    | &#x2705;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_2)   |
| 2024 | 3rd  | _Sarim Hashmi, et al._          | [Link](https://doi.org/10.48550/arXiv.2411.15872)    | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS24_3)   |
| 2023 | 1st  | _Andriy Myronenko, et al._      | [Link](https://arxiv.org/abs/2510.25058)             | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_1)   |
| 2023 | 2nd  | _Alyssa R Amod, et al._         | [Link](https://doi.org/10.1007/978-3-031-76163-8_22) | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_2)   |
| 2023 | 3rd  | _Ziyan Huang, et al._           | [Link](https://doi.org/10.1007/978-3-031-76163-8_13) | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.AfricaAlgorithms.BraTS23_3)   |

</details>

> Note: The SRI24 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in BraTS Africa Segmentation challenges.
<br>

#### Meningioma Segmentation
> Segmentation of Meningioma on brain MRI exams.
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import MeningiomaSegmenter
from brats.constants import MeningiomaAlgorithms

### Example for 2023 algorithms
segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS25_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation_23.nii.gz",
)
```

**Class:** `brats.MeningiomaSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.MeningiomaSegmenter))
<br>
**Challenge Paper 2023** [Link](https://doi.org/10.48550/arXiv.2305.07642)
<br>

| Year | Rank | Author                                 | Paper                                                      | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | -------------------------------------- | ---------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Yu Haitao, et al._                    | N/A                                                        | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS25_1) |
| 2025 | 2nd  | _Mohammad Mahdi Danesh Pajouh, et al._ | N/A                                                        | &#x274C;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS25_2) |
| 2023 | 1st  | _Andriy Myronenko, et al._             | [Link](https://arxiv.org/abs/2510.25058)                   | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Ziyan Huang, et al._                  | [Link](https://doi.org/10.1007/978-3-031-76163-8_13)       | &#x2705;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Daniel Capellán-Martín et al._        | [Link](https://api.semanticscholar.org/CorpusID:272599903) | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS23_3) |

</details>
<br>

#### Meningioma Radio Therapy Segmentation
> Segmentation of Meningioma on T1C brain MRI exams.
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>


```python
from brats import MeningiomaRTSegmenter
from brats.constants import MeningiomaRTAlgorithms

segmenter = MeningiomaRTSegmenter(algorithm=MeningiomaRTAlgorithms.BraTS25_1, cuda_devices="0")
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="segmentation_24.nii.gz",
)
```

**Class:** `brats.MeningiomaRTSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.MeningiomaRTSegmenter))
<br>
**Challenge Paper 2024** [Link](https://arxiv.org/abs/2405.18383)
<br>

| Year | Rank | Author                         | Paper                                       | CPU Support | Key Enum                                                                                                              |
| ---- | ---- | ------------------------------ | ------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Valeria Abramova, et al._     | N/A                                         | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaRTAlgorithms.BraTS25_1) |
| 2025 | 2nd  | _Sanskriti Srivastava, et al._ | N/A                                         | &#x2705;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaRTAlgorithms.BraTS25_2) |
| 2025 | 3rd  | _Nima Sadeghzadeh, et al._     | N/A                                         | &#x2705;    | [BraTS25_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaRTAlgorithms.BraTS25_3) |
| 2024 | 1st  | _Valeria Abramova_             | N/A                                         | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_1)   |
| 2024 | 2nd  | _Mehdi Astaraki_               | N/A                                         | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_2)   |
| 2024 | 3rd  | _Andre Ferreira, et al._       | [Link](https://arxiv.org/html/2411.04632v1) | &#x2705;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MeningiomaAlgorithms.BraTS24_3)   |


</details>

> Note: The MRI dataset in the Meningioma-Radiotherapy challenge was provided in native space. However,
the SRI24 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in BraTS Meningioma Pre-operative challenges.
<br>

#### Brain Metastases Segmentation
> Segmentation on brain metastases on MRI exams for pre- and post-treatment cases. 
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import MetastasesSegmenter
from brats.constants import MetastasesAlgorithms

segmenter = MetastasesSegmenter(algorithm=MetastasesAlgorithms.BraTS25_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)

```
> Note: If you're interested in Brain Metastases Segmentation, the [BrainLes AURORA package](https://github.com/BrainLesion/AURORA#aurora) may also be of interest.
<br>

**Class:** `brats.MetastasesSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.MetastasesSegmenter))
<br>
**Challenge Paper 2023** [Link](https://doi.org/10.48550/arXiv.2306.00838)
<br>
| Year | Rank | Author                     | Paper                                                | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | -------------------------- | ---------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Maria Bancerek, et al._   | N/A                                                  | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS25_1) |
| 2025 | 2nd  | _Wes Krikorian, et al._    | N/A                                                  | &#x2705;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS25_2) |
| 2023 | 1st  | _Andriy Myronenko, et al._ | [Link](https://arxiv.org/abs/2510.25058)             | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Siwei Yang, et al._       | [Link](https://doi.org/10.1007/978-3-031-76163-8_17) | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Ziyan Huang, et al._      | [Link](https://doi.org/10.1007/978-3-031-76163-8_13) | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MetastasesAlgorithms.BraTS23_3) |

</details>

> Note: The SRI24 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in BraTS Metastasis segmentation challenges.
<br>

#### Pediatric Segmentation
> Segmentation of pediatric brain tumors on MRI exams. 
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import PediatricSegmenter
from brats.constants import PediatricAlgorithms

segmenter = PediatricSegmenter(algorithm=PediatricAlgorithms.BraTS25_1, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```
> Note: If you're interested in Pediatric Segmentation, the [BrainLes PeTu package](https://github.com/BrainLesion/PeTu?tab=readme-ov-file#petu) may also be of interest.

**Class:** `brats.PediatricSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.PediatricSegmenter))
<br>
**Challenge Paper 2024** [Link](https://doi.org/10.48550/arXiv.2404.15009)
<br>
**Challenge Results Paper 2023** [Link](https://doi.org/10.48550/arXiv.2407.08855)
<br>
**Challenge Paper 2023** [Link](https://doi.org/10.48550/arXiv.2305.17033)
<br>

| Year | Rank | Author                           | Paper                                                      | CPU Support | Key Enum                                                                                                           |
| ---- | ---- | -------------------------------- | ---------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------ |
| 2025 | 1st  | _Yuxiao Yi, et al._              | N/A                                                        | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS25_1) |
| 2025 | 2nd  | _Meng-Yuan Chen, et al._         | N/A                                                        | &#x274C;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS25_2) |
| 2025 | 3rd  | _Haitao Yu, et al._              | N/A                                                        | &#x274C;    | [BraTS25_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS25_3) |
| 2024 | 1st  | _Mehdi Astaraki_                 | N/A                                                        | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Tim Mulvany, et al._            | [Link](https://doi.org/10.48550/arXiv.2410.14020)          | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Sarim Hashmi, et al._           | [Link](https://doi.org/10.48550/arXiv.2411.15872)          | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Daniel Capellán-Martín, et al._ | [Link](https://api.semanticscholar.org/CorpusID:272599903) | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_1) |
| 2023 | 2nd  | _Andriy Myronenko, et al._       | [Link](https://arxiv.org/abs/2510.25058)                   | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_2) |
| 2023 | 3rd  | _Yubo Zhou_                      | [Link](https://doi.org/10.1007/978-3-031-76163-8_5)        | &#x274C;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.PediatricAlgorithms.BraTS23_3) |


</details>

> Note: The SRI24 atlas, available on [Zenodo](https://zenodo.org/records/15927391), was employed for registration in BraTS Pediatric Tumor Segmentation challenges.
<br>

#### Generalizability Across Tumors (BraTS-GoAT) Segmentation 
> Segmentation algorithm, adapting and generalizing to different brain tumors with segmentation labels of different tumor sub-regions. 
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import GoATSegmenter
from brats.constants import GoATAlgorithms

segmenter = GoATSegmenter(algorithm=GoATAlgorithms.BraTS25_1A, cuda_devices="0")
# these parameters are optional, by default the latest winning algorithm will be used on cuda:0
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

**Class:** `brats.PediatricSegmenter` ([Docs](https://brats.readthedocs.io/en/latest/core/segmentation_algorithms.html#brats.core.segmentation_algorithms.PediatricSegmenter))
<br>
**Challenge Paper 2024:** N/A
<br> 

| Year | Rank | Author                      | Paper | CPU Support | Key Enum                                                                                                        |
| ---- | ---- | --------------------------- | ----- | ----------- | --------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Meng-Yuan Chen, et al._    | N/A   | &#x274C;    | [BraTS25_1A](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS25_1A) |
| 2025 | 1st  | _To-Liang Hsu, et al._      | N/A   | &#x274C;    | [BraTS25_1B](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS25_1B) |
| 2025 | 1st  | _Vaidehi Satushe, et al._   | N/A   | &#x274C;    | [BraTS25_1C](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS25_1C) |
| 2025 | 1st  | _Simone Bendazzoli, et al._ | N/A   | &#x274C;    | [BraTS25_1D](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS25_1D) |
| 2024 | 1st  | _Frank Miao, Shengjie Niu_  | N/A   | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.GoATAlgorithms.BraTS24_1)   |

</details>

> Note: The datasets used in this challenge were adapted from other segmentation challenges, so the atlas type depends on the original dataset.
<br>

### Inpainting Challenge 
<img src="https://github.com/BrainLesion/brats/blob/main/figures/inpainting_fig.png?raw=true" alt="matched_instance_figure" height="250"/>


> Algorithm to realistically synthesize and fill 3D healthy brain tissue in a region affected by glioma in brain MRI exams.  
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>


```python
from brats import Inpainter
from brats.constants import InpaintingAlgorithms

inpainter = Inpainter(algorithm=InpaintingAlgorithms.BraTS25_1A, cuda_devices="0")
inpainter.infer_single(
    t1n="path/to/voided_t1n.nii.gz",
    mask="path/to/mask.nii.gz",
    output_file="inpainting.nii.gz",
)
```

**Class:** `brats.Inpainter` ([Docs](https://brats.readthedocs.io/en/latest/core/inpainting_algorithms.html#brats.core.inpainting_algorithms.Inpainter))
<br>
**Challenge Paper 2023 and 2024** [Link](https://arxiv.org/pdf/2305.08992)
<br>
| Year | Rank | Author                   | Paper                                                | CPU Support | Key Enum                                                                                                              |
| ---- | ---- | ------------------------ | ---------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _Juexin Zhang, et al._   | N/A                                                  | &#x274C;    | [BraTS25_1A](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS25_1A) |
| 2025 | 1st  | _André Ferreira, et al._ | N/A                                                  | &#x274C;    | [BraTS25_1B](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS25_1B) |
| 2025 | 2nd  | _Juhyung Ha, et al._     | N/A                                                  | &#x274C;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS25_2)   |
| 2024 | 1st  | _Juexin Zhang et al._    | [Link](https://doi.org/10.48550/arXiv.2507.18126)    | &#x2705;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_1)   |
| 2024 | 2nd  | _André Ferreira, et al._ | [Link](https://arxiv.org/html/2411.04630v2)          | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_2)   |
| 2024 | 3rd  | _Alicia Durrer, et al._  | N/A                                                  | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS24_3)   |
| 2023 | 1st  | _Juexin Zhang, et al._   | [Link](https://doi.org/10.1007/978-3-031-76163-8_21) | &#x2705;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_1)   |
| 2023 | 2nd  | _Alicia Durrer, et al._  | [Link](https://doi.org/10.48550/arXiv.2402.17307)    | &#x274C;    | [BraTS23_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_2)   |
| 2023 | 3rd  | _Jiayu Huo, et al._      | [Link](https://doi.org/10.1007/978-3-031-76163-8_1)  | &#x2705;    | [BraTS23_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.InpaintingAlgorithms.BraTS23_3)   |

</details>

> Note: The datasets used in this challenge were adapted from other segmentation challenges, so the atlas type depends on the original dataset.
<br>


### Missing MRI Challenge
<img src="https://github.com/BrainLesion/brats/blob/main/figures/missingmri_fig.png?raw=true" alt="matched_instance_figure" height="250"/>


> Algorithm to realistically synthesize missing MRI modalities from available sequences to enhance brain tumor segmentation.  
<details>
<summary> Usage example (code) and top 3 participants </summary>
<br>

```python
from brats import MissingMRI
from brats.constants import MissingMRIAlgorithms

missing_mri = MissingMRI(algorithm=MissingMRIAlgorithms.BraTS25_1, cuda_devices="0")
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
<br>
**Challenge Paper 2024:** N/A
<br>
<br>
| Year | Rank | Author                                      | Paper                                             | CPU Support | Key Enum                                                                                                            |
| ---- | ---- | ------------------------------------------- | ------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------- |
| 2025 | 1st  | _André Ferreira, et al._                    | N/A                                               | &#x274C;    | [BraTS25_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS25_1) |
| 2025 | 2nd  | _Agustin Ujarky Cartaya Lathulerie, et al._ | N/A                                               | &#x274C;    | [BraTS25_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS25_2) |
| 2025 | 3rd  | _Lina Chator, et al._                       | N/A                                               | &#x2705;    | [BraTS25_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS25_3) |
| 2024 | 1st  | _Jihoon Cho et al._                         | [Link](https://arxiv.org/abs/2410.10269)          | &#x274C;    | [BraTS24_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_1) |
| 2024 | 2nd  | _Haowen Pang_                               | N/A                                               | &#x274C;    | [BraTS24_2](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_2) |
| 2024 | 3rd  | _Minjoo Lim et al._                         | [Link](https://doi.org/10.48550/arXiv.2502.19390) | &#x274C;    | [BraTS24_3](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS24_3) |
| 2023 | 1st  | _Ivo Baltruschat_                           | [Link](https://doi.org/10.48550/arXiv.2403.07800) | &#x274C;    | [BraTS23_1](https://brats.readthedocs.io/en/latest/utils/utils.html#brats.constants.MissingMRIAlgorithms.BraTS23_1) |

</details>

> Note: The datasets used in this challenge were adapted from other segmentation challenges, so the atlas type depends on the original dataset.
<br>

--- 

> [!TIP]
> For a full notebook example with more details, please check here:  
> [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/BrainLesion/tutorials/blob/main/BraTS/tutorial.ipynb)


## Data Preprocessing Requirements
BraTS challenge algorithms require preprocessed brain scans.  
This preprocessing typically includes co-registration, brain extraction, and registration to a challenge-specific brain atlas (template).  
Please refer to each challenge’s documentation for details on which template to use.  

For convenience, we provide challenge-specific wrappers around our [preprocessing package](https://github.com/BrainLesion/preprocessing), part of the [BrainLesion Suite](https://github.com/BrainLesion).  
You can install the package with the preprocessing extra using:  


```bash
pip install brats[preprocessing]
```

The modular architecture of the `BraTS Orchestrator` facilitates employing your own preprocessing routines.  
Alternatively, the [preprocessing package](https://github.com/BrainLesion/preprocessing) can be used independently to design custom preprocessing pipelines tailored to your specific needs.

## Citation

> [!IMPORTANT]
> If you use BraTS orchestrator in your research, please cite it to support the development!

Kofler, F., Rosier, M., Astaraki, M., Baid, U., Möller, H., Buchner, J. A., Steinbauer, F., Oswald, E., Rosa, E. de la, Ezhov, I., See, C. von, Kirschke, J., Schmick, A., Pati, S., Linardos, A., Pitarch, C., Adap, S., Rudie, J., Verdier, M. C. de, … Menze, B. (2025). BraTS orchestrator: Democratizing and Disseminating state-of-the-art brain tumor image analysis [arXiv preprint arXiv:2506.13807](https://doi.org/10.48550/arXiv.2506.13807)


```
@misc{kofler2025bratsorchestratordemocratizing,
      title={BraTS orchestrator : Democratizing and Disseminating state-of-the-art brain tumor image analysis}, 
      author={Florian Kofler and Marcel Rosier and Mehdi Astaraki and Ujjwal Baid and Hendrik Möller and Josef A. Buchner and Felix Steinbauer and Eva Oswald and Ezequiel de la Rosa and Ivan Ezhov and Constantin von See and Jan Kirschke and Anton Schmick and Sarthak Pati and Akis Linardos and Carla Pitarch and Sanyukta Adap and Jeffrey Rudie and Maria Correia de Verdier and Rachit Saluja and Evan Calabrese and Dominic LaBella and Mariam Aboian and Ahmed W. Moawad and Nazanin Maleki and Udunna Anazodo and Maruf Adewole and Marius George Linguraru and Anahita Fathi Kazerooni and Zhifan Jiang and Gian Marco Conte and Hongwei Li and Juan Eugenio Iglesias and Spyridon Bakas and Benedikt Wiestler and Marie Piraud and Bjoern Menze},
      year={2025},
      eprint={2506.13807},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.13807}, 
}
```

## Contributing

We welcome all kinds of contributions from the community!

### Reporting Bugs, Feature Requests, and Questions

Please open a new issue [here](https://github.com/BrainLesion/BraTS/issues).

### Code Contributions

Nice to have you on board! Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md) file.
