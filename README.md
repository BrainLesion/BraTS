# BraTS Orchestrator

[![Python Versions](https://img.shields.io/pypi/pyversions/brats)](https://pypi.org/project/brats/)
[![Stable Version](https://img.shields.io/pypi/v/brats?label=stable)](https://pypi.python.org/pypi/brats/)
[![Documentation Status](https://readthedocs.org/projects/brats/badge/?version=latest)](https://brats.readthedocs.io/?badge=latest)
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
\ \_/ / | | (__| | | |  __/\__ \ |_| | | (_| || (_) | |
 \___/|_|  \___|_| |_|\___||___/\__|_|  \__,_|\__\___/|_|
```

Providing the top-performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy-to-use Python API.

## Documentation

<table>
<tr>
<td align="center" width="50%">
<strong>Quickstart</strong><br>
Get up and running in minutes.<br>
<a href="https://brats.readthedocs.io/en/latest/getting-started/quickstart/">Go to Quickstart</a>
</td>
<td align="center" width="50%">
<strong>Segmentation</strong><br>
Adult glioma, meningioma, metastases, and more.<br>
<a href="https://brats.readthedocs.io/en/latest/guides/segmentation/">Go to Segmentation</a>
</td>
</tr>
<tr>
<td align="center" width="50%">
<strong>Inpainting</strong><br>
Synthesize healthy tissue in tumor regions.<br>
<a href="https://brats.readthedocs.io/en/latest/guides/inpainting/">Go to Inpainting</a>
</td>
<td align="center" width="50%">
<strong>Missing MRI</strong><br>
Generate missing MRI sequences.<br>
<a href="https://brats.readthedocs.io/en/latest/guides/missing-mri/">Go to Missing MRI</a>
</td>
</tr>
</table>

## Installation

```bash
pip install brats
pip install brats[preprocessing]
```

> [!IMPORTANT]
> BraTS Orchestrator requires **Docker** or **Singularity** to run algorithm containers. Most algorithms also need GPU support ([NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)). See the [installation docs](https://brats.readthedocs.io/en/latest/getting-started/installation/) for full setup instructions.

## Quick Example

```python
from brats import AdultGliomaPreAndPostTreatmentSegmenter

segmenter = AdultGliomaPreAndPostTreatmentSegmenter(cuda_devices="0")
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

## Citation

If you use BraTS Orchestrator in your research, please cite:

> Kofler, F., Rosier, M., et al. (2025). BraTS orchestrator: Democratizing and Disseminating state-of-the-art brain tumor image analysis. [arXiv:2506.13807](https://doi.org/10.48550/arXiv.2506.13807)

```bibtex
@misc{kofler2025bratsorchestratordemocratizing,
      title={BraTS orchestrator: Democratizing and Disseminating state-of-the-art brain tumor image analysis},
      author={Florian Kofler and Marcel Rosier et al.},
      year={2025},
      eprint={2506.13807},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2506.13807},
}
```

## Contributing

We welcome contributions! Please open a new issue [here](https://github.com/BrainLesion/BraTS/issues) or have a look at our [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache 2.0](LICENSE)
