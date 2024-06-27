from enum import Enum
from pathlib import Path


class Algorithms(str, Enum):
    """Parent class for constants of the available algorithms."""

    pass


class AdultGliomaAlgorithms(Algorithms):
    """Constants for the available adult glioma segmentation algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23 Adult Glioma Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Adult Glioma Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Adult Glioma Segmentation 3rd place (GPU only)"""


class MeningiomaAlgorithms(Algorithms):
    """Constants for the available meningioma segmentation algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23 Meningioma Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Meningioma Segmentation 2nd place (GPU and CPU)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Meningioma Segmentation 3rd place (GPU only)"""


class PediatricAlgorithms(Algorithms):
    """Constants for the available pediatric segmentation algorithms."""

    BraTS23_1 = "BraTS23_1"
    BraTS23_2 = "BraTS23_2"
    BraTS23_3 = "BraTS23_3"


# meta data file paths
ALGORITHM_DIR = Path(__file__).parent / "algorithms"
ADULT_GLIOMA_SEGMENTATION_ALGORITHMS = (
    ALGORITHM_DIR / "adult_glioma_segmentation_algorithms.yml"
)
MENINGIOMA_SEGMENTATION_ALGORITHMS = (
    ALGORITHM_DIR / "meningioma_segmentation_algorithms.yml"
)
PEDIATRIC_SEGMENTATION_ALGORITHMS = (
    ALGORITHM_DIR / "pediatric_segmentation_algorithms.yml"
)

# name formats
# All algorithms are designed to work with the following input file name format (validation set),
# hence all processed files will be mapped to the respective name space to ensure compatibility.
ADULT_GLIOMA_INPUT_NAME_SCHEMA = "BraTS-GLI-{id:05d}-000"
"""Adult Glioma input file name schema. E.g. BraTS-GLI-00001-000"""
MENINGIOMA_INPUT_NAME_SCHEMA = "BraTS-MEN-{id:05d}-000"
"""Meningioma input file name schema. E.g. BraTS-MEN-00001-000"""
PEDIATRIC_INPUT_NAME_SCHEMA = "BraTS-PED-{id:05d}-000"
"""Pediatric input file name schema. E.g. BraTS-PED-00001-000"""
