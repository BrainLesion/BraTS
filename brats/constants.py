from enum import Enum
from pathlib import Path

# TASK ENUM


class Backends(str, Enum):
    """Available backends for running the algorithms."""

    DOCKER = "docker"
    """Run the algorithms using Docker containers."""

    SINGULARITY = "singularity"
    """Run the algorithms using Singularity containers."""


class Task(str, Enum):
    """Available tasks."""

    SEGMENTATION = "SEGMENTATION"
    """Segmentation task."""

    INPAINTING = "INPAINTING"
    """Inpainting task."""

    MISSING_MRI = "MISSING_MRI"
    """Missing MRI task."""


# ALGORITHM ENUMS


class Algorithms(str, Enum):
    """Parent class for constants of the available algorithms."""

    pass


class AdultGliomaPreAndPostTreatmentAlgorithms(Algorithms):
    """Constants for the available adult glioma pre and post treatment segmentation algorithms."""

    BraTS25_1 = "BraTS25_1"
    """ BraTS25 Adult Glioma Segmentation 1st place """
    BraTS25_2 = "BraTS25_2"
    """ BraTS25 Adult Glioma Segmentation 2nd place """
    BraTS25_3A = "BraTS25_3A"
    """ BraTS25 Adult Glioma Segmentation 3rd place (tie) """
    BraTS25_3B = "BraTS25_3B"
    """ BraTS25 Adult Glioma Segmentation 3rd place (tie) """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24 Adult Glioma Segmentation 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24 Adult Glioma Segmentation 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24 Adult Glioma Segmentation 3rd place """


class AdultGliomaPreTreatmentAlgorithms(Algorithms):
    """Constants for the available adult glioma pre treatment segmentation algorithms."""

    BraTS23_1 = "BraTS23_1"
    """BraTS23 Adult Glioma Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Adult Glioma Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Adult Glioma Segmentation 3rd place (GPU only)"""


class MeningiomaRTAlgorithms(Algorithms):
    """Constants for the available meningioma segmentation - Radio Therapy algorithms."""

    BraTS25_1 = "BraTS25_1"
    """ BraTS25 Meningioma Segmentation 1st place """
    BraTS25_2 = "BraTS25_2"
    """ BraTS25 Meningioma Segmentation 2nd place """
    BraTS25_3 = "BraTS25_3"
    """ BraTS25 Meningioma Segmentation 3rd place """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24 Meningioma Segmentation 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24 Meningioma Segmentation 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24 Meningioma Segmentation 3rd place """


class MeningiomaAlgorithms(Algorithms):
    """Constants for the available meningioma segmentation algorithms."""

    BraTS25_1 = "BraTS25_1"
    """BraTS25 Meningioma Segmentation 1st place """
    BraTS25_2 = "BraTS25_2"
    """BraTS25 Meningioma Segmentation 2nd place """

    BraTS23_1 = "BraTS23_1"
    """BraTS23 Meningioma Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Meningioma Segmentation 2nd place (GPU and CPU)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Meningioma Segmentation 3rd place (GPU only)"""


class PediatricAlgorithms(Algorithms):
    """Constants for the available pediatric segmentation algorithms."""

    BraTS25_1 = "BraTS25_1"
    """ BraTS25 Pediatric Segmentation 1st place"""
    BraTS25_2 = "BraTS25_2"
    """ BraTS25 Pediatric Segmentation 2nd place"""
    BraTS25_3 = "BraTS25_3"
    """ BraTS25 Pediatric Segmentation 3rd place"""

    BraTS24_1 = "BraTS24_1"
    """ BraTS24 Pediatric Segmentation 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24 Pediatric Segmentation 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24 Pediatric Segmentation 3rd place """

    BraTS23_1 = "BraTS23_1"
    """BraTS23 Pediatric Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 Pediatric Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 Pediatric Segmentation 3rd place (GPU only)"""


class AfricaAlgorithms(Algorithms):
    """Constants for the available africa segmentation algorithms."""

    BraTS25_1 = "BraTS25_1"
    """ BraTS25 BraTS-SSA Segmentation 1st place """
    BraTS25_2A = "BraTS25_2A"
    """ BraTS25 BraTS-SSA Segmentation 2nd place (Tie)"""
    BraTS25_2B = "BraTS25_2B"
    """ BraTS25 BraTS-SSA Segmentation 2nd place (Tie) """
    BraTS25_3 = "BraTS25_3"
    """ BraTS25 BraTS-SSA Segmentation 3rd place """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24 BraTS-Africa Segmentation 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24 BraTS-Africa Segmentation 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24 BraTS-Africa Segmentation 3rd place """

    BraTS23_1 = "BraTS23_1"
    """BraTS23 BraTS-Africa Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23 BraTS-Africa Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23 BraTS-Africa Segmentation 3rd place (GPU and CPU)"""


class MetastasesAlgorithms(Algorithms):
    """Constants for the available Inpainting algorithms."""

    BraTS25_1 = "BraTS25_1"
    """BraTS25  Brain Metastases Segmentation 1st place"""
    BraTS25_2 = "BraTS25_2"
    """BraTS25  Brain Metastases Segmentation 2nd place"""

    BraTS23_1 = "BraTS23_1"
    """BraTS23  Brain Metastases Segmentation 1st place (GPU only)"""
    BraTS23_2 = "BraTS23_2"
    """BraTS23  Brain Metastases Segmentation 2nd place (GPU only)"""
    BraTS23_3 = "BraTS23_3"
    """BraTS23  Brain Metastases Segmentation 3rd place (GPU only)"""


class InpaintingAlgorithms(Algorithms):
    """Constants for the available BraTS Inpainting algorithms."""

    BraTS25_1A = "BraTS25_1A"
    """ BraTS25  Inpainting shared 1st place (tie) """
    BraTS25_1B = "BraTS25_1B"
    """ BraTS25  Inpainting shared 1st place (tie)"""
    BraTS25_2 = "BraTS25_2"
    """ BraTS25  Inpainting 2nd place """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24  Inpainting 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24  Inpainting 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24  Inpainting 3rd place """

    BraTS23_1 = "BraTS23_1"
    """ BraTS23  Inpainting 1st place """
    BraTS23_2 = "BraTS23_2"
    """ BraTS23  Inpainting 2nd place (Very Slow) """
    BraTS23_3 = "BraTS23_3"
    """ BraTS23  Inpainting 3rd place """


class MissingMRIAlgorithms(Algorithms):
    """Constants for the available missing mri algorithms."""

    BraTS25_1 = "BraTS25_1"
    """ BraTS25  MissingMRI 1st place """
    BraTS25_2 = "BraTS25_2"
    """ BraTS25  MissingMRI 2nd place """
    BraTS25_3 = "BraTS25_3"
    """ BraTS25  MissingMRI 3rd place """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24  MissingMRI 1st place """
    BraTS24_2 = "BraTS24_2"
    """ BraTS24  MissingMRI 2nd place """
    BraTS24_3 = "BraTS24_3"
    """ BraTS24  MissingMRI 3rd place """

    BraTS23_1 = "BraTS23_1"
    """ BraTS23  MissingMRI 1st place """


class GoATAlgorithms(Algorithms):
    """Constants for the available missing mri  algorithms."""

    BraTS25_1A = "BraTS25_1A"
    """ BraTS25 Generalizability Across Tumors (BraTS-GoAT) 1st place (tie) """
    BraTS25_1B = "BraTS25_1B"
    """ BraTS25 Generalizability Across Tumors (BraTS-GoAT) 1st place (tie) """
    BraTS25_1C = "BraTS25_1C"
    """ BraTS25 Generalizability Across Tumors (BraTS-GoAT) 1st place (tie) """
    BraTS25_1D = "BraTS25_1D"
    """ BraTS25 Generalizability Across Tumors (BraTS-GoAT) 1st place (tie) """

    BraTS24_1 = "BraTS24_1"
    """ BraTS24 Generalizability Across Tumors (BraTS-GoAT) 1st place (The only submission)"""


# DIRECTORIES
DATA_DIR = Path(__file__).parent / "data"
META_DIR = DATA_DIR / "meta"
PARAMETERS_DIR = DATA_DIR / "parameters"
ADDITIONAL_FILES_FOLDER = DATA_DIR / "additional_files"

# META DATA FILE PATHS
ADULT_GLIOMA_PRE_TREATMENT_SEGMENTATION_ALGORITHMS = (
    META_DIR / "adult_glioma_pre_treatment.yml"
)
ADULT_GLIOMA_PRE_AND_POST_TREATMENT_SEGMENTATION_ALGORITHMS = (
    META_DIR / "adult_glioma_pre_and_post_treatment.yml"
)
MENINGIOMA_SEGMENTATION_ALGORITHMS = META_DIR / "meningioma.yml"
MENINGIOMA_RT_SEGMENTATION_ALGORITHMS = META_DIR / "meningioma_rt.yml"
PEDIATRIC_SEGMENTATION_ALGORITHMS = META_DIR / "pediatric.yml"
AFRICA_SEGMENTATION_ALGORITHMS = META_DIR / "africa.yml"
METASTASES_SEGMENTATION_ALGORITHMS = META_DIR / "metastases.yml"
GOAT_SEGMENTATION_ALGORITHMS = META_DIR / "goat.yml"
INPAINTING_ALGORITHMS = META_DIR / "inpainting.yml"
MISSING_MRI_ALGORITHMS = META_DIR / "missing_mri.yml"

DUMMY_PARAMETERS = PARAMETERS_DIR / "dummy.yml"

# ZENODO
ZENODO_RECORD_BASE_URL = "https://zenodo.org/api/records"

OUTPUT_NAME_SCHEMA = {
    Task.SEGMENTATION: "{subject_id}.nii.gz",
    Task.INPAINTING: "{subject_id}-t1n-inference.nii.gz",
}


# PACKAGE citation
PACKAGE_CITATION = "https://arxiv.org/abs/2506.13807"
