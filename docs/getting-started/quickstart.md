# Quickstart

!!! important
    BraTS algorithms require **preprocessed** brain images (co-registered, skull-stripped, and atlas-registered).
    See the [preprocessing guide](../guides/preprocessing.md) for details.

## Segmentation

```python
from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms

segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
    algorithm=AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1,
    cuda_devices="0",
)
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

See the [segmentation guide](../guides/segmentation.md) for all supported segmentation tasks.

## Inpainting

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

See the [inpainting guide](../guides/inpainting.md) for details.

## Missing MRI Synthesis

```python
from brats import MissingMRI
from brats.constants import MissingMRIAlgorithms

missing_mri = MissingMRI(
    algorithm=MissingMRIAlgorithms.BraTS25_1,
    cuda_devices="0",
)
missing_mri.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    # t2f is missing — it will be synthesized
    t2w="path/to/t2w.nii.gz",
    output_file="inferred_t2f.nii.gz",
)
```

See the [missing MRI guide](../guides/missing-mri.md) for details.

## Next Steps

- :material-notebook: Walk through the full [Tutorial Notebook](../tutorials/tutorial.ipynb)
- :material-book-open-variant: Dive into the [Segmentation Guide](../guides/segmentation.md)
- :material-cog: Browse the [API Reference](../reference/api/core.md)
