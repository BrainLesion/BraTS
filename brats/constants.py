from enum import Enum


class AlgorithmKeys(str, Enum):
    BraTS23_yaziciz = "BraTS23_yaziciz"
    BraTS23_ferreira = "BraTS23_ferreira"


class Device(str, Enum):
    """Enum representing device for model inference."""

    CPU = "cpu"
    """Use CPU"""
    GPU = "cuda"
    """Use GPU (CUDA)"""
    AUTO = "auto"
    """Attempt to use GPU, fallback to CPU."""
