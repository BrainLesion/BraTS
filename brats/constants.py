from enum import Enum
from pathlib import Path


class AlgorithmKeys(str, Enum):
    pass


class AdultGliomaAlgorithmKeys(AlgorithmKeys):
    BraTS23_glioma_faking_it = "BraTS23_glioma_faking_it"
    """BraTS23 Adult Glioma Segmentation 1st place (GPU only)"""
    BraTS23_glioma_nvauto = "BraTS23_glioma_nvauto"
    """BraTS23 Adult Glioma Segmentation 2nd place (GPU only)"""
    BraTS23_glioma_biomedmbz = "BraTS23_glioma_biomedmbz"
    """BraTS23 Adult Glioma Segmentation 3rd place (GPU only)"""


class MeningiomaAlgorithmKeys(AlgorithmKeys):
    BraTS23_meningioma_nvauto = "BraTS23_meningioma_nvauto"
    """BraTS23 Meningioma Segmentation 1st place (GPU only)"""
    BraTS23_meningioma_blackbean = "BraTS23_meningioma_blackbean"
    """BraTS23 Meningioma Segmentation 2nd place (GPU and CPU)"""
    BraTS23_meningioma_CNMC_PMI2023 = "BraTS23_meningioma_CNMC_PMI2023"
    """BraTS23 Meningioma Segmentation 3rd place (GPU only)"""


# meta data file paths
PACKAGE_DIR = Path(__file__).parent / "algorithms"
ADULT_GLIOMA_SEGMENTATION_ALGORITHMS = (
    PACKAGE_DIR / "adult_glioma_segmentation_algorithms.yml"
)
MENINGIOMA_SEGMENTATION_ALGORITHMS = (
    PACKAGE_DIR / "meningioma_segmentation_algorithms.yml"
)

# name formats
# All algorithms are designed to work with the following input file name format (validation set),
# hence all processed files will be mapped to the respective name space to ensure compatibility.
ADULT_GLIOMA_INPUT_NAME_SCHEMA = "BraTS-GLI-{id:05d}-000"
MENINGIOMA_INPUT_NAME_SCHEMA = "BraTS-MEN-{id:05d}-000"
