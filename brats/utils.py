import shutil
from pathlib import Path


def standardize_subject_inputs(
    data_folder: Path,
    subject_id: str,
    t1c: Path | str,
    t1n: Path | str,
    t2f: Path | str,
    t2w: Path | str,
):
    """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@subject_id.
        Meaning, e.g. for adult glioma:
            BraTS-GLI-00000-000 \n
            ┣ BraTS-GLI-00000-000-t1c.nii.gz \n
            ┣ BraTS-GLI-00000-000-t1n.nii.gz \n
            ┣ BraTS-GLI-00000-000-t2f.nii.gz \n
            ┗ BraTS-GLI-00000-000-t2w.nii.gz \n

    Args:
        data_folder (Path): Parent folder where the subject folder will be created
        subject_id (str): Subject ID to be used for the folder and filenames
        t1c (Path | str): T1c image path
        t1n (Path | str): T1n image path
        t2f (Path | str): T2f image path
        t2w (Path | str): T2w image path
    """
    subject_folder = data_folder / subject_id
    subject_folder.mkdir(parents=True, exist_ok=True)

    # os.symlink would be more efficient but can cause issues on windows
    # TODO: use symlink on unix systems
    shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
    shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
    shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
    shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")
