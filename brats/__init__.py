from loguru import logger

from brats.core.inpainting_algorithms import Inpainter
from brats.core.missing_mri_algorithms import MissingMRI
from brats.core.segmentation_algorithms import (
    AdultGliomaPostTreatmentSegmenter,
    AdultGliomaPreTreatmentSegmenter,
    AfricaSegmenter,
    GoATSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
    PediatricSegmenter,
)

logger.remove()
