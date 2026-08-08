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
\ \_/ / | | (__| | | |  __/\__ \ |_| | | (_| || (_) | |
 \___/|_|  \___|_| |_|\___||___/\__|_|  \__,_|\__\___/|_|
```

Providing the top-performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy-to-use Python API powered by Docker or Singularity.

## Installation

```bash
pip install brats
# with preprocessing support
pip install brats[preprocessing]
```

> [!IMPORTANT]
> BraTS Orchestrator requires [Docker](https://docs.docker.com/get-docker/) and, for most algorithms, [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) is also supported.

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

## Documentation

- **[Full Documentation](https://brats.readthedocs.io)** — installation, quickstart, guides, and API reference
- **[Tutorial Notebook](https://brats.readthedocs.io/en/latest/tutorials/tutorial/)** — interactive Jupyter notebook with full examples

## Supported Tasks

BraTS Orchestrator provides access to top-performing algorithms for:

| Task | Description |
|------|-------------|
| **Segmentation** | 10 challenges: adult glioma (pre/post), Africa, meningioma, meningioma RT, metastases, pediatric, GoAT |
| **Inpainting** | Synthesizing healthy brain tissue in tumor-affected regions |
| **Missing MRI** | Synthesizing missing MRI modalities from available sequences |

See the [documentation](https://brats.readthedocs.io) for usage guides and the full algorithm tables.

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
