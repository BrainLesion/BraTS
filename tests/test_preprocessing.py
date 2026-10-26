import importlib
import sys
from contextlib import ExitStack
from types import ModuleType
from typing import ClassVar
from unittest.mock import MagicMock, patch

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

    class _RecordingModality:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class CenterModality(_RecordingModality):
        pass

    class Modality(_RecordingModality):
        pass

    class Normalizer:
        pass

    class _RecordingPreprocessor:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.run = MagicMock()
            type(self).instances.append(self)

    class AtlasCentricPreprocessor(_RecordingPreprocessor):
        pass

    class NativeSpacePreprocessor(_RecordingPreprocessor):
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


def test_coreg_native_space_bet_builds_native_preprocessor(preprocessing_module):
    paths = _all_modalities()
    preprocessing_module._coreg_native_space_bet(**paths)

    (preprocessor,) = preprocessing_module.NativeSpacePreprocessor.instances
    preprocessor.run.assert_called_once_with()

    center = preprocessor.kwargs["center_modality"]
    assert center.kwargs["modality_name"] == "t1c"
    assert center.kwargs["input_path"] == paths["t1c_input"]
    assert center.kwargs["raw_bet_output_path"] == paths["t1c_output"]
    assert center.kwargs["normalizer"] is None
    assert center.kwargs["atlas_correction"] is False

    moving = {
        item.kwargs["modality_name"]: item.kwargs
        for item in preprocessor.kwargs["moving_modalities"]
    }
    assert set(moving) == {"t1", "t2", "flair"}
    for name in ("t1", "t2", "flair"):
        assert moving[name]["raw_bet_output_path"] == paths[f"{name}_output"]
        assert moving[name]["normalizer"] is None


def test_coreg_native_space_bet_uses_normalized_outputs_with_normalizer(
    preprocessing_module,
):
    paths = _all_modalities()
    normalizer = preprocessing_module.Normalizer()
    preprocessing_module._coreg_native_space_bet(**paths, normalizer=normalizer)

    (preprocessor,) = preprocessing_module.NativeSpacePreprocessor.instances
    center = preprocessor.kwargs["center_modality"]
    assert center.kwargs["normalizer"] is normalizer
    assert center.kwargs["normalized_bet_output_path"] == paths["t1c_output"]
    assert "raw_bet_output_path" not in center.kwargs

    for item in preprocessor.kwargs["moving_modalities"]:
        assert item.kwargs["normalizer"] is normalizer
        assert "normalized_bet_output_path" in item.kwargs
        assert "raw_bet_output_path" not in item.kwargs


def test_coreg_atlasreg_bet_uses_normalized_outputs_with_normalizer(
    preprocessing_module,
):
    paths = _all_modalities()
    normalizer = preprocessing_module.Normalizer()
    preprocessing_module._coreg_atlasreg_bet(
        atlas=preprocessing_module.Atlas.BRATS_SRI24,
        **paths,
        normalizer=normalizer,
    )

    (preprocessor,) = preprocessing_module.AtlasCentricPreprocessor.instances
    center = preprocessor.kwargs["center_modality"]
    assert center.kwargs["normalizer"] is normalizer
    assert center.kwargs["normalized_bet_output_path"] == paths["t1c_output"]
    assert "raw_bet_output_path" not in center.kwargs

    for item in preprocessor.kwargs["moving_modalities"]:
        assert item.kwargs["normalizer"] is normalizer
        assert "normalized_bet_output_path" in item.kwargs
        assert "raw_bet_output_path" not in item.kwargs


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
        (MetastasesAlgorithms.BraTS25_1, "preprocess_coreg_native_space_bet"),
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
        "preprocess_coreg_native_space_bet",
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
