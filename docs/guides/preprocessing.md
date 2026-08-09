# Preprocessing

BraTS challenge algorithms require **preprocessed** brain scans. This typically includes:

- Co-registration of multi-modal MRI sequences
- Brain extraction (skull stripping)
- Registration to a challenge-specific brain atlas (SRI24 or MNI152)

## Atlases

| Atlas | Used by | Available at |
|-------|---------|-------------|
| **SRI24** | Pre-2024 Glioma Pre-Treatment, Africa, Meningioma, Metastases, Pediatric | [Zenodo](https://zenodo.org/records/15927391) |
| **MNI152** | 2024+ Glioma Post-Treatment, Meningioma RT | [Zenodo](https://zenodo.org/records/15927391) |

The required atlas is also noted in each algorithm's table in the [algorithm reference](../reference/algorithms.md).

## Using the Built-in Preprocessing

Install the preprocessing extra:

```bash
pip install brats[preprocessing]
# or: uv add 'brats[preprocessing]'
```

The `brats.preprocessing` module provides convenience wrappers around the [preprocessing package](https://github.com/BrainLesion/preprocessing):

```python
from brats.preprocessing import preprocess_coreg_sri24reg_bet

preprocess_coreg_sri24reg_bet(
    input_dir="path/to/raw/data",
    output_dir="path/to/preprocessed/data",
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
| `preprocess_for_challenge` | Router that picks the right pipeline for a given challenge |

## Custom Preprocessing

The modular architecture of BraTS Orchestrator allows you to use your own preprocessing routines. The [preprocessing package](https://github.com/BrainLesion/preprocessing) can also be used independently to design custom pipelines tailored to your specific needs.
