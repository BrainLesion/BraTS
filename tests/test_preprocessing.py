import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from brats.constants import MetastasesAlgorithms


@pytest.fixture
def preprocessing_module(monkeypatch):
    brainles_preprocessing = ModuleType("brainles_preprocessing")
    constants = ModuleType("brainles_preprocessing.constants")
    modality = ModuleType("brainles_preprocessing.modality")
    normalization = ModuleType("brainles_preprocessing.normalization")
    preprocessor = ModuleType("brainles_preprocessing.preprocessor")

    class Atlas:
        BRATS_MNI152 = "MNI152"
        BRATS_SRI24 = "SRI24"

    class CenterModality:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Modality:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Normalizer:
        pass

    class AtlasCentricPreprocessor:
        pass

    class NativeSpacePreprocessor:
        pass

    constants.Atlas = Atlas
    modality.CenterModality = CenterModality
    modality.Modality = Modality
    normalization.Normalizer = Normalizer
    preprocessor.AtlasCentricPreprocessor = AtlasCentricPreprocessor
    preprocessor.NativeSpacePreprocessor = NativeSpacePreprocessor

    for module in [
        brainles_preprocessing,
        constants,
        modality,
        normalization,
        preprocessor,
    ]:
        monkeypatch.setitem(sys.modules, module.__name__, module)

    sys.modules.pop("brats.preprocessing", None)
    module = importlib.import_module("brats.preprocessing")
    yield module
    sys.modules.pop("brats.preprocessing", None)


def _all_modalities() -> dict[str, str]:
    return {
        "t1_input": "t1.nii.gz",
        "t1c_input": "t1c.nii.gz",
        "t2_input": "t2.nii.gz",
        "flair_input": "flair.nii.gz",
        "t1_output": "t1-out.nii.gz",
        "t1c_output": "t1c-out.nii.gz",
        "t2_output": "t2-out.nii.gz",
        "flair_output": "flair-out.nii.gz",
    }


def test_native_space_wrapper_delegates_to_native_pipeline(preprocessing_module):
    with patch.object(preprocessing_module, "_coreg_native_space_bet") as mock_pipeline:
        preprocessing_module.preprocess_coreg_native_space_bet(**_all_modalities())

    mock_pipeline.assert_called_once_with(
        **_all_modalities(),
        normalizer=None,
    )


def test_2025_metastases_use_native_space_pipeline(preprocessing_module):
    with (
        patch.object(
            preprocessing_module, "preprocess_coreg_native_space_bet"
        ) as mock_native,
        patch.object(
            preprocessing_module, "preprocess_coreg_sri24reg_bet"
        ) as mock_sri24,
    ):
        preprocessing_module.preprocess_for_challenge(
            MetastasesAlgorithms.BraTS25_1,
            **_all_modalities(),
        )

    mock_native.assert_called_once_with(
        *_all_modalities().values(),
        normalizer=None,
    )
    mock_sri24.assert_not_called()


def test_2023_metastases_use_sri24_pipeline(preprocessing_module):
    with (
        patch.object(
            preprocessing_module, "preprocess_coreg_native_space_bet"
        ) as mock_native,
        patch.object(
            preprocessing_module, "preprocess_coreg_sri24reg_bet"
        ) as mock_sri24,
    ):
        preprocessing_module.preprocess_for_challenge(
            MetastasesAlgorithms.BraTS23_1,
            **_all_modalities(),
        )

    mock_sri24.assert_called_once_with(
        *_all_modalities().values(),
        normalizer=None,
    )
    mock_native.assert_not_called()
