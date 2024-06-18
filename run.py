from pathlib import Path
from brats.inferer import AdultGliomaInferer, MeningiomaInferer
from brats.constants import AdultGliomaAlgorithmKeys, MeningiomaAlgorithmKeys

# inferer = AdultGliomaInferer(algorithm=AdultGliomaAlgorithmKeys.BraTS23_glioma_nvauto, cuda_devices="1")
# # base = Path("/home/marcel/Projects/helmholtz/brats/workspace/data/single")
# base = Path("/home/marcelrosier/brats_data/adult_glioma/BraTS-GLI-00001-000/")
# inferer.infer_single(
#     t1c=base / "t1c.nii.gz",
#     t1n=base / "t1n.nii.gz",
#     t2f=base / "t2f.nii.gz",
#     t2w=base / "t2w.nii.gz",
#     output_file="single_out/nvauto_seg.nii.gz",
# )

alg = MeningiomaAlgorithmKeys.BraTS23_meningioma_CNMC_PMI2023
inferer = MeningiomaInferer(algorithm=alg, cuda_devices="1")

base = Path("/home/marcelrosier/brats_data/adult_meningioma/BraTS-MEN-00000-000")
inferer.infer_single(
    t1c=base / "BraTS-MEN-00000-000-t1c.nii.gz",
    t1n=base / "BraTS-MEN-00000-000-t1n.nii.gz",
    t2f=base / "BraTS-MEN-00000-000-t2f.nii.gz",
    t2w=base / "BraTS-MEN-00000-000-t2w.nii.gz",
    output_file=f"single_out/seg-{alg.value}.nii.gz",
)
