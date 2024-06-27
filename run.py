from pathlib import Path
from brats.inferer import AdultGliomaInferer, MeningiomaInferer, PediatricInferer
from brats.constants import (
    AdultGliomaAlgorithms,
    MeningiomaAlgorithms,
    PediatricAlgorithms,
)


# alg = AdultGliomaAlgorithmKeys.BraTS23_glioma_faking_it
# inferer = AdultGliomaInferer(algorithm=alg, cuda_devices="0")
# base = Path("/home/marcelrosier/brats_data/adult_glioma/BraTS-GLI-00001-000/")
# inferer.infer_single(
#     t1c=base / "t1c.nii.gz",
#     t1n=base / "t1n.nii.gz",
#     t2f=base / "t2f.nii.gz",
#     t2w=base / "t2w.nii.gz",
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


# pediatric
alg = PediatricAlgorithms.BraTS23_1
inferer = PediatricInferer(algorithm=alg, cuda_devices="0")

base = Path("/home/marcelrosier/brats_data/pediatric/BraTS-PED-00030-000")
inferer.infer_single(
    t1c=base / "BraTS-PED-00030-000-t1c.nii.gz",
    t1n=base / "BraTS-PED-00030-000-t1n.nii.gz",
    t2f=base / "BraTS-PED-00030-000-t2f.nii.gz",
    t2w=base / "BraTS-PED-00030-000-t2w.nii.gz",
    output_file=f"single_out/seg-{alg.value}.nii.gz",
)
