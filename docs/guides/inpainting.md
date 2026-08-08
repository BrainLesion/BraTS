# Inpainting

![Inpainting example](https://github.com/BrainLesion/brats/blob/main/figures/inpainting_fig.png?raw=true)

Inpainting algorithms realistically synthesize and fill 3D healthy brain tissue in regions affected by glioma in brain MRI exams.

## Usage

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

## Available Algorithms

--8<-- "algorithm-tables/inpainting.md"
