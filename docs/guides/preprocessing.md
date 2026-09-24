# Preprocessing

BraTS challenge algorithms require **preprocessed** brain scans. This typically includes:

- Co-registration of multi-modal MRI sequences
- Brain extraction (skull stripping)
- Registration to a challenge-specific brain atlas (SRI24 or MNI152)

## Data Preprocessing Specifications

!!! important
    **Voxel Intensity Preservation:** All MRI volumes across all challenge tracks retain their post-preprocessing spatial intensity values. **No global voxel intensity normalization** (e.g., z-score standardization or min-max scaling) has been pre-applied to the dataset files. Models expect raw intensity inputs as processed by the official BraTS pipeline.

---

### Overview of Preprocessing Standards

Depending on the specific challenge task and edition year, input multi-parametric MRI (mpMRI) sequences undergo specific spatial registration and other preprocessing steps (e.g., skull-stripping vs. defacing):

| Challenge Task | Modalities | Reference Space | Anonymization / Masking | Notes / Timeline |
| :--- | :--- | :--- | :--- | :--- |
| **Adult Glioma (Pre-Treatment)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | SRI24 | Skull-stripped | BraTS 2023 and earlier. |
| **Adult Glioma (Pre & Post-Treatment)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | MNI152 | Skull-stripped | BraTS 2024+ track. |
| **Brain Metastases (MET)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | SRI24 (BraTS 2023)<br>Native Space (BraTS 2025+ additions) | Skull-stripped | BraTS 2023 cases aligned to SRI24. BraTS 2025+ additions (pre & post-treatment) are in native space, co-registered to `T1c`. |
| **Meningioma (Pre-Operative)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | SRI24 | Skull-stripped | Standard pre-operative diagnostic imaging. |
| **Meningioma (Radiotherapy RT)** | Single sequence (`T1c`) | Native Space | Defaced | Treatment planning sequence; defaced rather than skull-stripped. |
| **Pediatric Brain Tumors (PED)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | SRI24 | Defaced | Defaced rather than skull-stripped. |
| **Sub-Saharan Africa Glioma (SSA)** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | SRI24 | Skull-stripped | Standardized on SRI24 template. |
| **Cross-Tumor Generalizability** | 4 mpMRI (`T1c`, `T1n`, `T2f`, `T2w`) | Depends on the source dataset | Depends on the source dataset | Combined dataset spanning adult/SSA glioma, pediatric, pre-op meningioma, and brain metastases. |

---

### Per-Task Technical Details

#### 1. Adult Glioma Segmentation (Pre-Treatment)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:**
  BraTS 2023 and earlier pre-treatment exams are co-registered and spatially
  normalized to the **SRI24 atlas space**, followed by rigid skull-stripping.

#### 2. Adult Glioma Segmentation (Pre & Post-Treatment)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:** BraTS 2024+ exams are co-registered and template-matched to **MNI152 atlas space** with skull-stripping applied across all exams.

#### 3. Brain Metastases Segmentation (Pre & Post-Treatment)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:**
  * **BraTS 2023 Cohorts:** Pre-treatment exams registered to **SRI24 atlas space** and skull-stripped.
  * **BraTS 2025+ Cohorts:** Expanded to include both pre- and post-treatment cases provided in **Native Space** (co-registered directly to the `T1c` acquisition volume) with skull-stripping applied.

#### 4. Pre-Operative Meningioma Segmentation

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:** Diagnostic co-registered mpMRI volumes aligned to **SRI24 atlas space** and skull-stripped.

#### 5. Radiotherapy Planning Meningioma Segmentation (Meningioma-RT)

* **Modalities:** Single sequence contrast-enhanced T1-weighted (`T1c`) MRI.
* **Spatial Alignment & Anonymization:** Provided in **Native Space** (no template registration) and processed using **defacing** rather than skull-stripping.

#### 6. Pediatric Brain Tumor Segmentation (PED)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:** Registered to **SRI24 atlas space**. Uses **defacing** algorithms rather than skull-stripping.

#### 7. Sub-Saharan Africa Glioma Segmentation (SSA)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Spatial Alignment & Anonymization:** Standardized to **SRI24 atlas space** with complete skull-stripping.

#### 8. Cross-Tumor Generalizability (BraTS-GoAT)

* **Modalities:** 4 mpMRI sequences (`T1c`, `T1n`, `T2f`, `T2w`).
* **Composition:** Aggregated multi-site dataset combining Adult Glioma, Sub-Saharan Africa Glioma, Pediatric Tumors, Pre-operative Meningioma, and Brain Metastases.
* **Spatial Alignment & Anonymization:** The atlas and anonymization method follow the source dataset. Do not assume that every GoAT input uses SRI24; use the preprocessing requirements of the source challenge.

!!! warning "BraTS-GoAT preprocessing"
    `preprocess_for_challenge` cannot infer the source dataset behind a GoAT input.
    Confirm the source challenge's preprocessing requirements and use custom
    preprocessing when they differ from the default SRI24 pipeline. Source-aware
    GoAT preprocessing will require a follow-up task.

## Atlases

| Atlas | Used by | Available at |
|-------|---------|-------------|
| **SRI24** | Pre-2024 Glioma Pre-Treatment, Africa, Meningioma, BraTS 2023 Metastases, Pediatric | [Zenodo](https://zenodo.org/records/15927391) |
| **MNI152** | 2024+ Adult Glioma (Pre & Post-Treatment) | [Zenodo](https://zenodo.org/records/15927391) |

Meningioma RT data remains in native image space and uses defacing only; it does not require an atlas.

The required atlas is also noted in each algorithm's table in the [algorithm reference](../reference/algorithms.md).

## Using the Built-in Preprocessing

Install the preprocessing extra:

```bash
pip install brats[preprocessing]
```

The `brats.preprocessing` module provides convenience wrappers around the [preprocessing package](https://github.com/BrainLesion/preprocessing). The wrappers accept individual input and output image paths, not input/output directories:

```python
from brats.preprocessing import preprocess_coreg_sri24reg_bet

preprocess_coreg_sri24reg_bet(
    t1_input="path/to/raw/t1.nii.gz",
    t1c_input="path/to/raw/t1c.nii.gz",
    t2_input="path/to/raw/t2.nii.gz",
    flair_input="path/to/raw/flair.nii.gz",
    t1_output="path/to/preprocessed/t1.nii.gz",
    t1c_output="path/to/preprocessed/t1c.nii.gz",
    t2_output="path/to/preprocessed/t2.nii.gz",
    flair_output="path/to/preprocessed/flair.nii.gz",
)
```

Available wrappers:

| Function | Description |
|---|---|
| `preprocess_coreg_sri24reg_bet` | Co-registration + SRI24 atlas registration + brain extraction |
| `preprocess_coreg_sri24reg_defacing` | Co-registration + SRI24 atlas registration + defacing |
| `preprocess_coreg_mni152reg_bet` | Co-registration + MNI152 atlas registration + brain extraction |
| `preprocess_coreg_native_space_bet` | Co-registration in native space + brain extraction |
| `preprocess_coreg_sri24reg_bet_allow_missing` | Same as above but tolerates missing modalities |
| `preprocess_deface_only` | Defacing only |
| `preprocess_for_challenge` | Router that picks the right pipeline for a given challenge; source-dataset-specific GoAT, inpainting, and Missing MRI atlas choices may require custom preprocessing |

## Custom Preprocessing

The modular architecture of BraTS Orchestrator allows you to use your own preprocessing routines. The [preprocessing package](https://github.com/BrainLesion/preprocessing) can also be used independently to design custom pipelines tailored to your specific needs.
