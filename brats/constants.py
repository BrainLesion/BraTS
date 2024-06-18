from enum import Enum
from pathlib import Path


class AdultGliomaAlgorithmKeys(str, Enum):
    BraTS23_glioma_faking_it = "BraTS23_glioma_faking_it"
    BraTS23_glioma_nvauto = "BraTS23_glioma_nvauto"
    BraTS23_glioma_biomedmbz = "BraTS23_glioma_biomedmbz"
    
class MeningiomaAlgorithmKeys(str, Enum):
    BraTS23_meningioma_nvauto = "BraTS23_meningioma_nvauto"


class Device(str, Enum):
    """Enum representing device for model inference."""

    CPU = "cpu"
    """Use CPU"""
    GPU = "cuda"
    """Use GPU (CUDA)"""
    AUTO = "auto"
    """Attempt to use GPU, fallback to CPU."""



# meta data file paths
PACKAGE_DIR = Path(__file__).parent / "algorithms"
ADULT_GLIOMA_SEGMENTATION_ALGORITHMS = PACKAGE_DIR / "adult_glioma_segmentation_algorithms.yml"
MENINGIOMA_SEGMENTATION_ALGORITHMS = PACKAGE_DIR / "meningioma_segmentation_algorithms.yml"

BRATS_INPUT_NAME_SCHEMA = "BraTS-GLI-{id:05d}-000"
