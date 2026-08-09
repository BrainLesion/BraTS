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

BraTS Orchestrator provides the top-performing algorithms from the Brain Tumor Segmentation (BraTS) challenges, through an easy-to-use Python API backed by containers.

## Quick Install

```bash
pip install brats
```

## Quick Example

```python
from brats import AdultGliomaPreAndPostTreatmentSegmenter

segmenter = AdultGliomaPreAndPostTreatmentSegmenter()
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

## Explore

<div class="grid cards" markdown>

- :material-rocket-launch: **Quickstart**

    Get up and running in minutes with a minimal example.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

- :material-brain: **Using BraTS**

    Segmentation, inpainting, and missing MRI synthesis — pick your task.

    [:octicons-arrow-right-24: Using BraTS](guides/segmentation.md)

- :material-notebook: **Tutorial**

    Interactive Jupyter notebook with full examples.

    [:octicons-arrow-right-24: Tutorial](tutorials/tutorial.ipynb)

- :material-cog: **API Reference**

    Auto-generated API documentation from source code.

    [:octicons-arrow-right-24: API Reference](reference/api/core.md)

</div>

---

!!! important "Citation"
    If you use BraTS Orchestrator in your research, please [cite it](citation.md).
