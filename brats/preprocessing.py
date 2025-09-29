from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from brats.constants import Algorithms

try:
    from brainles_preprocessing.modality import Modality, CenterModality
    from brainles_preprocessing.preprocessor import (
        AtlasCentricPreprocessor,
        NativeSpacePreprocessor,
    )
    from brainles_preprocessing.constants import Atlas
except ImportError as e:
    raise ImportError(
        "The `brainles_preprocessing` extra is required for preprocessing tasks, please ensure you installed it via `pip install brats[preprocessing]`."
    ) from e


# class BraTSPreprocessor:
#     """BraTS specific preprocessing functions."""

#     def __init__(self, challenge: Algorithms) -> None:
#         self.challenge = challenge

#     def preprocess(self):


def _coreg_atlasreg_bet(
    atlas: Atlas,
    t1_input: Optional[Union[str, Path]] = None,
    t1c_input: Optional[Union[str, Path]] = None,
    t2_input: Optional[Union[str, Path]] = None,
    flair_input: Optional[Union[str, Path]] = None,
    t1_output: Optional[Union[str, Path]] = None,
    t1c_output: Optional[Union[str, Path]] = None,
    t2_output: Optional[Union[str, Path]] = None,
    flair_output: Optional[Union[str, Path]] = None,
    allow_missing: bool = False,
) -> None:
    # Build modality mapping in a cleaner way
    modality_pairs = [
        ("t1", t1_input, t1_output),
        ("t1c", t1c_input, t1c_output),
        ("t2", t2_input, t2_output),
        ("flair", flair_input, flair_output),
    ]

    # Filter valid modalities (both input and output must be provided)
    valid_modalities = {
        name: {"input_path": inp, "raw_bet_output_path": out}
        for name, inp, out in modality_pairs
        if inp is not None and out is not None
    }

    # Validate minimum requirements
    min_required = 3 if allow_missing else 4
    if len(valid_modalities) < min_required:
        missing_desc = "at least three" if allow_missing else "all four"
        raise ValueError(
            f"Need {missing_desc} modalities with valid input/output paths"
        )

    # Choose center modality (prefer t1c, fallback to t1)
    center_name = "t1c" if "t1c" in valid_modalities else "t1"
    center_data = valid_modalities[center_name]

    center = CenterModality(
        modality_name=center_name,
        input_path=center_data["input_path"],
        raw_bet_output_path=center_data["raw_bet_output_path"],
        atlas_correction=False,
    )

    # Create moving modalities (all except center)
    moving_modalities = [
        Modality(
            modality_name=name,
            input_path=data["input_path"],
            raw_bet_output_path=data["raw_bet_output_path"],
            atlas_correction=False,
        )
        for name, data in valid_modalities.items()
        if name != center_name
    ]

    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
        atlas_image_path=atlas,
    )

    preprocessor.run()


def _coreg_atlasreg_deface(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
    atlas: Atlas,
) -> None:

    center = CenterModality(
        modality_name="t1c",
        input_path=t1c_input,
        raw_defaced_output_path=t1c_output,
        atlas_correction=False,
    )

    moving_modalities = [
        Modality(
            modality_name="t1",
            input_path=t1_input,
            raw_defaced_output_path=t1_output,
            atlas_correction=False,
        ),
        Modality(
            modality_name="t2",
            input_path=t2_input,
            raw_defaced_output_path=t2_output,
            atlas_correction=False,
        ),
        Modality(
            modality_name="flair",
            input_path=flair_input,
            raw_defaced_output_path=flair_output,
            atlas_correction=False,
        ),
    ]
    preprocessor = AtlasCentricPreprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
        atlas_image_path=atlas,
    )

    preprocessor.run()


def _coreg_bet(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
) -> None:

    center = CenterModality(
        modality_name="t1c",
        input_path=t1c_input,
        raw_bet_output_path=t1c_output,
        atlas_correction=False,
    )

    moving_modalities = [
        Modality(
            modality_name="t1",
            input_path=t1_input,
            raw_bet_output_path=t1_output,
            atlas_correction=False,
        ),
        Modality(
            modality_name="t2",
            input_path=t2_input,
            raw_bet_output_path=t2_output,
            atlas_correction=False,
        ),
        Modality(
            modality_name="flair",
            input_path=flair_input,
            raw_bet_output_path=flair_output,
            atlas_correction=False,
        ),
    ]
    preprocessor = NativeSpacePreprocessor(
        center_modality=center,
        moving_modalities=moving_modalities,
    )

    preprocessor.run()


#######
# Preprocessing functions for different BraTS challenges
#######


def preprocess_coreg_sri24reg_bet(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
) -> None:
    """t1, t1c, t2, flair to SRI24 with co-registration and BET (most segmentation challenges and inpainting)

    Args:
        t1_input (Union[str, Path]): Path to the input T1 image.
        t1c_input (Union[str, Path]): Path to the input T1c image
        t2_input (Union[str, Path]): Path to the input T2 image.
        flair_input (Union[str, Path]): Path to the input FLAIR image.
        t1_output (Union[str, Path]): Path to the output preprocessed T1 image
        t1c_output (Union[str, Path]): Path to the output preprocessed T1c image
        t2_output (Union[str, Path]): Path to the output preprocessed T2 image
        flair_output (Union[str, Path]): Path to the output preprocessed FLAIR image

    Returns:
        None

    """
    _coreg_atlasreg_bet(
        t1_input=t1_input,
        t1c_input=t1c_input,
        t2_input=t2_input,
        flair_input=flair_input,
        t1_output=t1_output,
        t1c_output=t1c_output,
        t2_output=t2_output,
        flair_output=flair_output,
        atlas=Atlas.BRATS_SRI24,
    )


def preprocess_coreg_sri24reg_defacing(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
) -> None:
    """t1, t1c, t2, flair to SRI24 with co-registration and defacing (pediatric challenge)

    Args:
        t1_input (Union[str, Path]): Path to the input T1 image.
        t1c_input (Union[str, Path]): Path to the input T1c image
        t2_input (Union[str, Path]): Path to the input T2 image.
        flair_input (Union[str, Path]): Path to the input FLAIR image.
        t1_output (Union[str, Path]): Path to the output preprocessed T1 image
        t1c_output (Union[str, Path]): Path to the output preprocessed T1c image
        t2_output (Union[str, Path]): Path to the output preprocessed T2 image
        flair_output (Union[str, Path]): Path to the output preprocessed FLAIR image

    Returns:
        None

    """
    _coreg_atlasreg_deface(
        t1_input=t1_input,
        t1c_input=t1c_input,
        t2_input=t2_input,
        flair_input=flair_input,
        t1_output=t1_output,
        t1c_output=t1c_output,
        t2_output=t2_output,
        flair_output=flair_output,
        atlas=Atlas.BRATS_SRI24,
    )


def preprocess_coreg_mni152reg_bet(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
) -> None:
    """t1, t1c, t2, flair to MNI152 with co-registration and BET (GLI Post)

    Args:
        t1_input (Union[str, Path]): Path to the input T1 image.
        t1c_input (Union[str, Path]): Path to the input T1c image
        t2_input (Union[str, Path]): Path to the input T2 image.
        flair_input (Union[str, Path]): Path to the input FLAIR image.
        t1_output (Union[str, Path]): Path to the output preprocessed T1 image
        t1c_output (Union[str, Path]): Path to the output preprocessed T1c image
        t2_output (Union[str, Path]): Path to the output preprocessed T2 image
        flair_output (Union[str, Path]): Path to the output preprocessed FLAIR image

    Returns:
        None

    """
    _coreg_atlasreg_bet(
        t1_input=t1_input,
        t1c_input=t1c_input,
        t2_input=t2_input,
        flair_input=flair_input,
        t1_output=t1_output,
        t1c_output=t1c_output,
        t2_output=t2_output,
        flair_output=flair_output,
        atlas=Atlas.BRATS_MNI152,
    )


def preprocess_coreg_bet(
    t1_input: Union[str, Path],
    t1c_input: Union[str, Path],
    t2_input: Union[str, Path],
    flair_input: Union[str, Path],
    t1_output: Union[str, Path],
    t1c_output: Union[str, Path],
    t2_output: Union[str, Path],
    flair_output: Union[str, Path],
) -> None:
    _coreg_bet(
        t1_input=t1_input,
        t1c_input=t1c_input,
        t2_input=t2_input,
        flair_input=flair_input,
        t1_output=t1_output,
        t1c_output=t1c_output,
        t2_output=t2_output,
        flair_output=flair_output,
    )


def preprocess_coreg_sri24reg_bet_allow_missing(
    t1_input: Optional[Union[str, Path]] = None,
    t1c_input: Optional[Union[str, Path]] = None,
    t2_input: Optional[Union[str, Path]] = None,
    flair_input: Optional[Union[str, Path]] = None,
    t1_output: Optional[Union[str, Path]] = None,
    t1c_output: Optional[Union[str, Path]] = None,
    t2_output: Optional[Union[str, Path]] = None,
    flair_output: Optional[Union[str, Path]] = None,
) -> None:
    """t1, t1c, t2, flair to SRI24 with co-registration and BET while allowing one missing modality (missing MRI challenge)

    Args:
        t1_input (Optional[Union[str, Path]]): Path to the input T1 image.
        t1c_input (Optional[Union[str, Path]]): Path to the input T1c image
        t2_input (Optional[Union[str, Path]]): Path to the input T2 image.
        flair_input (Optional[Union[str, Path]]): Path to the input FLAIR image.
        t1_output (Optional[Union[str, Path]]): Path to the output preprocessed T1 image
        t1c_output (Optional[Union[str, Path]]): Path to the output preprocessed T1c image
        t2_output (Optional[Union[str, Path]]): Path to the output preprocessed T2 image
        flair_output (Optional[Union[str, Path]]): Path to the output preprocessed FLAIR image

    Returns:
        None

    """
    _coreg_atlasreg_bet(
        t1_input=t1_input,
        t1c_input=t1c_input,
        t2_input=t2_input,
        flair_input=flair_input,
        t1_output=t1_output,
        t1c_output=t1c_output,
        t2_output=t2_output,
        flair_output=flair_output,
        atlas=Atlas.BRATS_SRI24,
        allow_missing=True,
    )


def preprocess_for_challenge(
    challenge: Algorithms,
    t1_input: Optional[Union[str, Path]] = None,
    t1c_input: Optional[Union[str, Path]] = None,
    t2_input: Optional[Union[str, Path]] = None,
    flair_input: Optional[Union[str, Path]] = None,
    t1_output: Optional[Union[str, Path]] = None,
    t1c_output: Optional[Union[str, Path]] = None,
    t2_output: Optional[Union[str, Path]] = None,
    flair_output: Optional[Union[str, Path]] = None,
) -> None:
    """Automatically select the correct preprocessing pipeline for a specific BraTS challenge.

    Args:
        challenge (Algorithms): The BraTS challenge algorithm enum
        t1_input to flair_output: Input and output paths for each modality

    Raises:
        ValueError: If required modalities are missing for the challenge
    """
    challenge_name = str(challenge)

    # Helper to validate all modalities are present
    def _require_all_modalities():
        all_paths = [
            t1_input,
            t1c_input,
            t2_input,
            flair_input,
            t1_output,
            t1c_output,
            t2_output,
            flair_output,
        ]
        if any(path is None for path in all_paths):
            raise ValueError(
                f"All modalities required for {challenge_name} preprocessing"
            )
        return all_paths  # Type checker knows these are not None after the check

    # Route to appropriate preprocessing function
    if "AdultGliomaPreAndPostTreatment" in challenge_name:
        paths = _require_all_modalities()
        preprocess_coreg_mni152reg_bet(*paths)  # type: ignore

    elif "Pediatric" in challenge_name:
        paths = _require_all_modalities()
        preprocess_coreg_sri24reg_defacing(*paths)  # type: ignore

    elif "MissingMRI" in challenge_name:
        preprocess_coreg_sri24reg_bet_allow_missing(
            t1_input,
            t1c_input,
            t2_input,
            flair_input,
            t1_output,
            t1c_output,
            t2_output,
            flair_output,
        )

    else:  # Most challenges use SRI24 with BET
        paths = _require_all_modalities()
        preprocess_coreg_sri24reg_bet(*paths)  # type: ignore
