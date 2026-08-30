from loguru import logger

from brats.core.inpainting_algorithms import Inpainter
from brats.core.missing_mri_algorithms import MissingMRI
from brats.core.segmentation_algorithms import (
    AdultGliomaPreAndPostTreatmentSegmenter,
    AdultGliomaPreTreatmentSegmenter,
    AfricaSegmenter,
    GoATSegmenter,
    MeningiomaRTSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
    PediatricSegmenter,
)

logger.disable("brats")
