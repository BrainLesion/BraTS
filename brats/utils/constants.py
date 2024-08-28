from enum import Enum
from pathlib import Path

# TASK ENUM


class Task(str, Enum):
    """Available tasks."""

    SEGMENTATION = "SEGMENTATION"
    """Segmentation task."""

    INPAINTING = "INPAINTING"
    """Inpainting task."""


# ALGORITHM ENUMS


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
    """BraTS23 Pediatric Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Pediatric Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Pediatric Segmentation 3rd place (GPU only)"""


class AfricaAlgorithms(Algorithms):
    """Constants for the available africa segmentation algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23 BraTS-Africa Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 BraTS-Africa Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 BraTS-Africa Segmentation 3rd place (GPU and CPU)"""


class MetastasesAlgorithms(Algorithms):
    """Constants for the available Inpainting algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23  Brain Metastases Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23  Brain Metastases Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23  Brain Metastases Segmentation 3rd place (GPU only)"""


class InpaintingAlgorithms(Algorithms):
    """Constants for the available BraTS Inpainting algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23  Inpainting 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23  Inpainting 2nd place (GPU only) (Very Slow)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23  Inpainting 3rd place (GPU only)"""


# DIRECTORIES
DATA_DIR = Path(__file__).parent.parent / "data"
META_DIR = DATA_DIR / "meta"
PARAMETERS_DIR = DATA_DIR / "parameters"
ADDITIONAL_FILES_FOLDER = DATA_DIR / "additional_files"

# META DATA FILE PATHS
ADULT_GLIOMA_SEGMENTATION_ALGORITHMS = META_DIR / "adult_glioma.yml"
MENINGIOMA_SEGMENTATION_ALGORITHMS = META_DIR / "meningioma.yml"
PEDIATRIC_SEGMENTATION_ALGORITHMS = META_DIR / "pediatric.yml"
AFRICA_SEGMENTATION_ALGORITHMS = META_DIR / "africa.yml"
METASTASES_SEGMENTATION_ALGORITHMS = META_DIR / "metastases.yml"
INPAINTING_ALGORITHMS = META_DIR / "inpainting.yml"

DUMMY_PARAMETERS = PARAMETERS_DIR / "dummy.yml"

# ZENODO
ZENODO_RECORD_BASE_URL = "https://zenodo.org/api/records"

OUTPUT_NAME_SCHEMA = {
    Task.SEGMENTATION: "{subject_id}.nii.gz",
    Task.INPAINTING: "{subject_id}-t1n-inference.nii.gz",
}
