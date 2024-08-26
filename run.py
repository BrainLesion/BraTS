from pathlib import Path
from brats import (
    AdultGliomaSegmenter,
    MeningiomaSegmenter,
    PediatricSegmenter,
    AfricaSegmenter,
)
from brats.constants import (
    AdultGliomaAlgorithms,
    MeningiomaAlgorithms,
    PediatricAlgorithms,
    AfricaAlgorithms,
)


alg = AfricaAlgorithms.BraTS23_1
segmenter = AfricaSegmenter(algorithm=alg, cuda_devices="0")
base = Path("/home/marcelrosier/brats_data/africa/BraTS-SSA-00126-000")
segmenter.infer_single(
    t1c=base / "BraTS-SSA-00126-000-t1c.nii.gz",
    t1n=base / "BraTS-SSA-00126-000-t1n.nii.gz",
    t2f=base / "BraTS-SSA-00126-000-t2f.nii.gz",
    t2w=base / "BraTS-SSA-00126-000-t2w.nii.gz",
    output_file=f"africa_out/seg-{alg.value}.nii.gz",
    log_file=f"africa_out/log-{alg.value}.log",
)

# alg = AdultGliomaAlgorithms.BraTS23_1
# inferer = AdultGliomaSegmenter(algorithm=alg, cuda_devices="0")

# base = Path("/home/marcelrosier/brats_data/adult_glioma/BraTS-GLI-00001-000")
# inferer.infer_single(
#     t1c=base / "t1c.nii.gz",
#     t1n=base / "t1n.nii.gz",
#     t2f=base / "t2f.nii.gz",
#     t2w=base / "t2w.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
#     log_file=f"single_out/log-{alg.value}.log",
# )
# base = Path("/home/marcelrosier/brats_data/oslo/16236606/day_0000")
# inferer.infer_single(
#     t1c=base / "_t1c.nii.gz",
#     t1n=base / "_t1.nii.gz",
#     t2f=base / "_fla.nii.gz",
#     t2w=base / "_t2.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
#     log_file=f"single_out/oslo-log-{alg.value}.log",
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
# alg = PediatricAlgorithms.BraTS23_3
# segmenter = PediatricSegmenter(algorithm=alg, cuda_devices="4")

# base = Path("/home/ivan_marcel/test_data/PED/BraTS-PED-00030-000")
# segmenter.infer_single(
#     t1c=base / "BraTS-PED-00030-000-t1c.nii.gz",
#     t1n=base / "BraTS-PED-00030-000-t1n.nii.gz",
#     t2f=base / "BraTS-PED-00030-000-t2f.nii.gz",
#     t2w=base / "BraTS-PED-00030-000-t2w.nii.gz",
#     output_file=f"single_out/seg-{alg.value}.nii.gz",
#     log_file=f"single_out/log-{alg.value}.txt",
# )
# segmenter.infer_batch(
#     data_folder=base.parent,
#     output_folder=Path("batch_out"),
#     log_file=Path("batch_out/log.log"),
# )
