from loguru import logger

from brats.core.inpainting_algorithms import Inpainter
from brats.core.missing_mri_algorithms import MissingMRI
from brats.core.segmentation_algorithms import (
    AdultGliomaPreTreatmentSegmenter,
    AdultGliomaPreAndPostTreatmentSegmenter,
    AfricaSegmenter,
    GoATSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
    MeningiomaRTSegmenter,
    PediatricSegmenter,
)

logger.remove()
