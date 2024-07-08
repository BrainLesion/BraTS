from pathlib import Path
from brats import AdultGliomaSegmenter, MeningiomaSegmenter, PediatricSegmenter
from brats.constants import (
    AdultGliomaAlgorithms,
    MeningiomaAlgorithms,
    PediatricAlgorithms,
)


# alg = AdultGliomaAlgorithms.BraTS23_1
# inferer = AdultGliomaInferer(algorithm=alg, cuda_devices="4")
# base = Path("/home/ivan_marcel/test_data/GLI/BraTS-GLI-00001-000/")
# inferer.infer_single(
#     t1c=base / "BraTS-GLI-00001-000-t1c.nii.gz",
#     t1n=base / "BraTS-GLI-00001-000-t1n.nii.gz",
#     t2f=base / "BraTS-GLI-00001-000-t2f.nii.gz",
#     t2w=base / "BraTS-GLI-00001-000-t2w.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
# )
# import time

# start = time.time()
# alg = MeningiomaAlgorithms.BraTS23_2
# inferer = MeningiomaInferer(algorithm=alg, cuda_devices="0")

# base = Path("/home/marcelrosier/brats_data/adult_meningioma/BraTS-MEN-00000-000")
# inferer.infer_single(
#     t1c=base / "BraTS-MEN-00000-000-t1c.nii.gz",
#     t1n=base / "BraTS-MEN-00000-000-t1n.nii.gz",
#     t2f=base / "BraTS-MEN-00000-000-t2f.nii.gz",
#     t2w=base / "BraTS-MEN-00000-000-t2w.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
# )
# print("Took: ", time.time() - start)


# # pediatric
alg = PediatricAlgorithms.BraTS23_3
segmenter = PediatricSegmenter(algorithm=alg, cuda_devices="4")

base = Path("/home/ivan_marcel/test_data/PED/BraTS-PED-00030-000")
# segmenter.infer_single(
#     t1c=base / "BraTS-PED-00030-000-t1c.nii.gz",
#     t1n=base / "BraTS-PED-00030-000-t1n.nii.gz",
#     t2f=base / "BraTS-PED-00030-000-t2f.nii.gz",
#     t2w=base / "BraTS-PED-00030-000-t2w.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
#     log_file=f"single_out/log-{alg.value}.txt",
# )
segmenter.infer_batch(
    data_folder=base.parent,
    output_folder=Path("batch_out"),
    log_file=Path("batch_out/log.log"),
)
