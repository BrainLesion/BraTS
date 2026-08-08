# Segmentation

![Segmentation example](https://github.com/BrainLesion/brats/blob/main/figures/segmentation_fig.png?raw=true)

BraTS Orchestrator provides access to top-performing segmentation algorithms from multiple BraTS challenges.

## Adult Glioma (Pre & Post-Treatment)

Segmentation on pre- and post-treatment brain MRI exams (4 modalities: T1c, T1n, T2f, T2w).

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

--8<-- "algorithm-tables/adult-glioma-pre-post.md"

---

## Adult Glioma (Pre-Treatment)

Pre-treatment glioma segmentation (4 modalities).

```python
from brats import AdultGliomaPreTreatmentSegmenter
from brats.constants import AdultGliomaPreTreatmentAlgorithms

segmenter = AdultGliomaPreTreatmentSegmenter(
    algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
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

--8<-- "algorithm-tables/adult-glioma-pre-treatment.md"

---

## BraTS-Africa

Glioma segmentation on Sub-Saharan African patient population MRI exams (4 modalities).

```python
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

segmenter = AfricaSegmenter(
    algorithm=AfricaAlgorithms.BraTS25_1,
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

--8<-- "algorithm-tables/africa.md"

---

## Meningioma

Meningioma segmentation on brain MRI exams (4 modalities).

```python
from brats import MeningiomaSegmenter
from brats.constants import MeningiomaAlgorithms

segmenter = MeningiomaSegmenter(
    algorithm=MeningiomaAlgorithms.BraTS25_1,
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

--8<-- "algorithm-tables/meningioma.md"

---

## Meningioma Radio Therapy

Segmentation on T1c-only MRI exams.

```python
from brats import MeningiomaRTSegmenter
from brats.constants import MeningiomaRTAlgorithms

segmenter = MeningiomaRTSegmenter(
    algorithm=MeningiomaRTAlgorithms.BraTS25_1,
    cuda_devices="0",
)
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    output_file="segmentation.nii.gz",
)
```

--8<-- "algorithm-tables/meningioma-rt.md"

---

## Brain Metastases

Brain metastases segmentation for pre- and post-treatment cases (4 modalities).

```python
from brats import MetastasesSegmenter
from brats.constants import MetastasesAlgorithms

segmenter = MetastasesSegmenter(
    algorithm=MetastasesAlgorithms.BraTS25_1,
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

--8<-- "algorithm-tables/metastases.md"

---

## Pediatric

Pediatric brain tumor segmentation (4 modalities).

```python
from brats import PediatricSegmenter
from brats.constants import PediatricAlgorithms

segmenter = PediatricSegmenter(
    algorithm=PediatricAlgorithms.BraTS25_1,
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

--8<-- "algorithm-tables/pediatric.md"

---

## BraTS-GoAT

Segmentation algorithm that adapts and generalizes to different brain tumors with varying label definitions (4 modalities).

```python
from brats import GoATSegmenter
from brats.constants import GoATAlgorithms

segmenter = GoATSegmenter(
    algorithm=GoATAlgorithms.BraTS25_1A,
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

--8<-- "algorithm-tables/goat.md"

---

!!! note "Legacy Algorithms"
    Some legacy segmentation algorithms from BraTS challenges before 2023 are available via [BraTS Toolkit](https://github.com/neuronflow/BraTS-Toolkit).
