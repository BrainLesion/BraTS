# Missing MRI

![Missing MRI example](https://github.com/BrainLesion/brats/blob/main/figures/missingmri_fig.png?raw=true)

Synthesize a missing MRI modality from the three available sequences.

## Single Subject

Provide any three of the four standard modalities (T1c, T1n, T2f, T2w) — the missing one will be synthesized:

```python
from brats import MissingMRI
from brats.constants import MissingMRIAlgorithms

missing_mri = MissingMRI(
    algorithm=MissingMRIAlgorithms.BraTS25_1,
    cuda_devices="0",
)

# Synthesize the missing T2f modality
missing_mri.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    # t2f="path/to/t2f.nii.gz",  # omitted — will be synthesized
    t2w="path/to/t2w.nii.gz",
    output_file="inferred_t2f.nii.gz",
)
```

## Batch Processing

```python
missing_mri.infer_batch(
    data_folder="path/to/subjects/",
    output_folder="path/to/output/",
)
```

## Available Algorithms

--8<-- "algorithm-tables/missing-mri.md"
