# Glossary

Domain terminology for the BraTS orchestrator.

## MRI Modalities

| Term | Definition |
|------|------------|
| **t1c** (T1c, T1C) | T1-weighted MRI with gadolinium contrast enhancement |
| **t1n** (T1, t1) | T1-weighted MRI without contrast (native) |
| **t2f** (FLAIR) | T2-weighted Fluid-Attenuated Inversion Recovery MRI |
| **t2w** (T2, t2) | T2-weighted MRI |
| **mask** | Binary mask indicating the region to inpaint |
| **t1n-voided** | T1n image with the tumor region artificially removed; input for inpainting |

## Brain Atlases

| Term | Definition |
|------|------------|
| **SRI24** | MRI-based brain atlas used for spatial registration in 2023 and most pre-2024 challenges |
| **MNI152** | Montreal Neurological Institute 152 template; used for 2024+ Glioma Post-treatment challenges |

## Algorithm Naming

Algorithms are named using the pattern `BraTS{YY}_{rank}`, where `YY` is the challenge year (e.g., 25) and `rank` is the placement (1 = 1st place). Suffix letters (`3A`, `3B`, `1A`, `1B`) indicate ties or multiple winning entries.

## Backends

| Term | Definition |
|------|------------|
| **Docker** | Default backend. Algorithms run in Docker containers via `docker-py`. |
| **Singularity** | Alternative backend for HPC environments that prohibit Docker. Converts Docker images to Singularity sandbox format. |

## Container Convention

| Term | Definition |
|------|------------|
| **mlcube** | A container interoperability specification ([mlcommons.org/mlcube](https://mlcommons.org/mlcube)) that uses numbered I/O directories (`mlcube_io0` for data, `mlcube_io1` for weights, `mlcube_io2` for output, `mlcube_io3` for parameters). Used by BraTS algorithms from year ≤ 2024. Superseded in 2025+ by native `/input` and `/output` volume mounts. |

## Challenge Types

### Adult Glioma Pre-Treatment Segmentation

Segmentation of glioma on pre-treatment brain MRI. Uses 4 modalities (t1c, t1n, t2f, t2w), registered to SRI24 atlas. Covers BraTS 2023 challenge year.

### Adult Glioma Pre and Post-Treatment Segmentation

Segmentation of glioma on both pre- and post-treatment MRI. Uses 4 modalities, registered to MNI152 atlas. Covers BraTS 2024 and 2025 challenge years.

### BraTS-Africa (SSA) Segmentation

Glioma segmentation on Sub-Saharan Africa patient populations. Uses 4 modalities, registered to SRI24 atlas.

### Meningioma Segmentation

Segmentation of meningioma — typically benign tumors arising from the brain covering membranes (meninges). Uses 4 modalities, registered to SRI24 atlas.

### Meningioma Radio Therapy (RT) Segmentation

Segmentation of meningioma from radiotherapy-planning MRI. T1C-only modality. Data is in native space (no atlas registration).

### Brain Metastases Segmentation

Segmentation of brain metastases — secondary tumors that have spread to the brain from cancers elsewhere in the body. Uses 4 modalities, registered to SRI24 atlas.

### Pediatric Segmentation

Segmentation of pediatric brain tumors. Uses 4 modalities, registered to SRI24 atlas. Preprocessing uses defacing instead of skull-stripping for privacy.

### Generalizability Across Tumors (BraTS-GoAT)

Segmentation across different tumor types, testing generalization to unseen tumor categories. Uses 4 modalities. Atlas depends on the source dataset (adapted from other segmentation challenges).

### Inpainting

Synthesize healthy brain tissue in a glioma-affected region. Input: voided T1n (t1n-voided) + binary mask. Output: restored T1n image. Atlas depends on the source dataset.

### Missing MRI (BraSyn)

Synthesize a missing MRI modality from 3 available modalities. Input: any 3 of t1c, t1n, t2f, t2w. Output: the missing 4th modality. Atlas depends on the source dataset. Formal challenge name: Brain MRI Synthesis Challenge (BraSyn).

## Preprocessing

| Term | Definition |
|------|------------|
| **BET** (Brain Extraction) | Skull-stripping: removing non-brain tissue from MRI images. Used in most segmentation challenges. |
| **Defacing** | Removing facial features from brain MRI for privacy protection. Used in the pediatric challenge instead of BET. |
| **Co-registration** | Aligning multiple MRI modalities of the same subject to a common spatial reference frame. |
| **Atlas registration** | Warping a subject's brain MRI to a standard atlas template (SRI24 or MNI152). |
| **Native space** | The original (non-registered) image coordinate space. Meningioma RT data is in native space and does not undergo atlas registration. |
