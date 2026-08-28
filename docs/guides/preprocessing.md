# Preprocessing

BraTS challenge algorithms require **preprocessed** brain scans. This typically includes:

- Co-registration of multi-modal MRI sequences
- Brain extraction (skull stripping)
- Registration to a challenge-specific brain atlas (SRI24 or MNI152)

## Atlases

| Atlas | Used by | Available at |
|-------|---------|-------------|
| **SRI24** | Pre-2024 Glioma Pre-Treatment, Africa, Meningioma, Metastases, Pediatric | [Zenodo](https://zenodo.org/records/15927391) |
| **MNI152** | 2024+ Glioma Post-Treatment | [Zenodo](https://zenodo.org/records/15927391) |

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
| `preprocess_coreg_sri24reg_bet_allow_missing` | Same as above but tolerates missing modalities |
| `preprocess_deface_only` | Defacing only |
| `preprocess_for_challenge` | Router that picks the right pipeline for a given challenge; source-dataset-specific GoAT, inpainting, and Missing MRI atlas choices may require custom preprocessing |

## Custom Preprocessing

The modular architecture of BraTS Orchestrator allows you to use your own preprocessing routines. The [preprocessing package](https://github.com/BrainLesion/preprocessing) can also be used independently to design custom pipelines tailored to your specific needs.
