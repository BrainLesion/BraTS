# Inpainting

![Inpainting example](https://github.com/BrainLesion/brats/blob/main/figures/inpainting_fig.png?raw=true)

Synthesize healthy brain tissue in tumor-affected regions of brain MRI exams.

## Single Subject

```python
from brats import Inpainter
from brats.constants import InpaintingAlgorithms

inpainter = Inpainter(
    algorithm=InpaintingAlgorithms.BraTS25_1A,
    cuda_devices="0",
)
inpainter.infer_single(
    t1n="path/to/voided_t1n.nii.gz",
    mask="path/to/mask.nii.gz",
    output_file="inpainting.nii.gz",
)
```

## Batch Processing

```python
inpainter.infer_batch(
    data_folder="path/to/subjects/",
    output_folder="path/to/output/",
)
```

## Available Algorithms

--8<-- "algorithm-tables/inpainting.md"
