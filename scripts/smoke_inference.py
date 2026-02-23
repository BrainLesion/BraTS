from pathlib import Path

from brats import AdultGliomaPreAndPostTreatmentSegmenter
from brats.constants import AdultGliomaPreAndPostTreatmentAlgorithms
from brats.constants import Backends

def main():
    # TODO: point these to your preprocessed files
    case_dir = Path("data/preprocessed_case")

    t1c = case_dir / "t1c.nii.gz"
    t1n = case_dir / "t1n.nii.gz"
    t2f = case_dir / "t2f.nii.gz"
    t2w = case_dir / "t2w.nii.gz"

    out_file = Path("outputs/segmentation.nii.gz")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Pick a CPU-compatible algorithm to avoid GPU/toolkit setup for the smoke test
    segmenter = AdultGliomaPreAndPostTreatmentSegmenter(
        algorithm=AdultGliomaPreAndPostTreatmentAlgorithms.BraTS25_2,
        # do NOT set cuda_devices for CPU run
    )

    segmenter.infer_single(
        t1c=str(t1c),
        t1n=str(t1n),
        t2f=str(t2f),
        t2w=str(t2w),
        output_file=str(out_file),
        backend=Backends.DOCKER,
    )

    print("Wrote:", out_file)

if __name__ == "__main__":
    main()
