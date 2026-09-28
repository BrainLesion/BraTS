import importlib
import sys
from contextlib import ExitStack
from types import ModuleType
from unittest.mock import patch

import pytest

from brats.constants import (
    AdultGliomaPreAndPostTreatmentAlgorithms,
    AdultGliomaPreTreatmentAlgorithms,
    AfricaAlgorithms,
    GoATAlgorithms,
    InpaintingAlgorithms,
    MeningiomaAlgorithms,
    MeningiomaRTAlgorithms,
    MetastasesAlgorithms,
    MissingMRIAlgorithms,
    PediatricAlgorithms,
)


@pytest.fixture
def preprocessing_module(monkeypatch):
    """Import preprocessing with a lightweight fake optional dependency."""
    brainles_preprocessing = ModuleType("brainles_preprocessing")
    constants = ModuleType("brainles_preprocessing.constants")
    modality = ModuleType("brainles_preprocessing.modality")
    normalization = ModuleType("brainles_preprocessing.normalization")
    preprocessor = ModuleType("brainles_preprocessing.preprocessor")

    class Atlas:
        BRATS_MNI152 = "MNI152"
        BRATS_SRI24 = "SRI24"

    class CenterModality:
        pass

    class Modality:
        pass

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


@pytest.mark.parametrize(
    ("challenge", "expected_pipeline"),
    [
        (
            AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1,
            "preprocess_coreg_mni152reg_bet",
        ),
        (PediatricAlgorithms.BraTS25_1, "preprocess_coreg_sri24reg_defacing"),
        (
            MissingMRIAlgorithms.BraTS25_1,
            "preprocess_coreg_sri24reg_bet_allow_missing",
        ),
        (MeningiomaRTAlgorithms.BraTS25_1, "preprocess_deface_only"),
        (
            AdultGliomaPreTreatmentAlgorithms.BraTS23_1,
            "preprocess_coreg_sri24reg_bet",
        ),
        (AfricaAlgorithms.BraTS25_1, "preprocess_coreg_sri24reg_bet"),
        (GoATAlgorithms.BraTS25_1A, "preprocess_coreg_sri24reg_bet"),
        (InpaintingAlgorithms.BraTS25_1A, "preprocess_coreg_sri24reg_bet"),
        (MeningiomaAlgorithms.BraTS25_1, "preprocess_coreg_sri24reg_bet"),
        (MetastasesAlgorithms.BraTS23_1, "preprocess_coreg_sri24reg_bet"),
    ],
)
def test_preprocess_for_challenge_dispatches_by_enum(
    preprocessing_module, challenge, expected_pipeline
):
    pipeline_names = [
        "preprocess_coreg_mni152reg_bet",
        "preprocess_coreg_sri24reg_defacing",
        "preprocess_coreg_sri24reg_bet_allow_missing",
        "preprocess_deface_only",
        "preprocess_coreg_sri24reg_bet",
    ]
    paths = _all_modalities()

    with ExitStack() as stack:
        pipelines = {
            name: stack.enter_context(patch.object(preprocessing_module, name))
            for name in pipeline_names
        }
        preprocessing_module.preprocess_for_challenge(challenge, **paths)

    pipelines[expected_pipeline].assert_called_once()
    for name, pipeline in pipelines.items():
        if name != expected_pipeline:
            pipeline.assert_not_called()


def test_preprocess_for_challenge_rejects_raw_value(preprocessing_module):
    with pytest.raises(TypeError, match="supported Algorithms enum member"):
        preprocessing_module.preprocess_for_challenge(
            AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1.value,
            **_all_modalities(),
        )


def test_preprocess_for_challenge_validates_required_modalities(preprocessing_module):
    paths = _all_modalities()
    paths["t1_input"] = None

    with pytest.raises(ValueError, match="All modalities required"):
        preprocessing_module.preprocess_for_challenge(
            AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_1,
            **paths,
        )
