from enum import Enum


class AlgorithmKeys(str, Enum):
    BraTS23_faking_it = "BraTS23_faking_it"
    BraTS23_nvauto = "BraTS23_nvauto"
    BraTS23_biomedmbz = "BraTS23_biomedmbz"


class Device(str, Enum):
    """Enum representing device for model inference."""

    CPU = "cpu"
    """Use CPU"""
    GPU = "cuda"
    """Use GPU (CUDA)"""
    AUTO = "auto"
    """Attempt to use GPU, fallback to CPU."""


BRATS_INPUT_NAME_SCHEMA = "BraTS-GLI-{id:05d}-000"
