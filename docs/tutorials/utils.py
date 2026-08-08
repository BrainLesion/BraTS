from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

DATA_FOLDER = Path("data")


def visualize_segmentation_data(
    data_folder: Union[str, Path] = DATA_FOLDER,
    subject_id: str = "BraTS-GLI-00001-000",
    slice_index: int = 75,
):
    """Visualize the MRI modalities for a given slice index

    Args:
        data_folder (Union[str, Path], optional): Path to the folder containing the t1, t1c, t2 & flair file. Defaults to DATA_FOLDER.
        slice_index (int, optional): Slice to be visualized (first index in data of shape (155, 240, 240)). Defaults to 75.
    """
    _, axes = plt.subplots(1, 4, figsize=(12, 10))

    subject_path = Path(data_folder) / subject_id
    modalities = ["t1n", "t1c", "t2f", "t2w"]
    for i, mod in enumerate(modalities):
        modality_file = subject_path / f"{subject_id}-{mod}.nii.gz"
        modality_np = nib.load(modality_file).get_fdata().transpose(2, 1, 0)
        axes[i].set_title(mod)
        axes[i].imshow(modality_np[slice_index, :, :], cmap="gray")
        axes[i].axis("off")


def visualize_inpainting_data(
    data_folder: Union[str, Path] = DATA_FOLDER,
    subject_id: str = "BraTS-GLI-00001-000",
    slice_index: int = 75,
):
    """Visualize the MRI modalities for a given slice index

    Args:
        data_folder (Union[str, Path], optional): Path to the folder containing the t1n and mask files. Defaults to DATA_FOLDER.
        slice_index (int, optional): Slice to be visualized (first index in data of shape (155, 240, 240)). Defaults to 75.
    """
    _, axes = plt.subplots(1, 2, figsize=(6, 10))

    subject_path = Path(data_folder) / subject_id
    modalities = ["t1n-voided", "mask"]
    for i, mod in enumerate(modalities):
        modality_file = subject_path / f"{subject_id}-{mod}.nii.gz"
        modality_np = nib.load(modality_file).get_fdata().transpose(2, 1, 0)
        axes[i].set_title(mod)
        axes[i].imshow(modality_np[slice_index, :, :], cmap="gray")
        axes[i].axis("off")


def visualize_segmentation(
    modality_file: Union[str, Path], segmentation_file: Union[str, Path]
):
    """Visualize the MRI modality and the segmentation

    Args:
        modality_file (Union[str, Path]): Path to the desired modality file
        segmentation_file (Union[str, Path]): Path to the segmentation file
    """
    modality_np = nib.load(modality_file).get_fdata().transpose(2, 1, 0)
    seg_np = nib.load(segmentation_file).get_fdata().transpose(2, 1, 0)

    _, ax = plt.subplots(1, 2, figsize=(8, 4))

    slice_index = modality_np.shape[0] // 2  # You can choose any slice here

    # Mask out background (0) in the segmentation
    seg_slice = seg_np[slice_index, :, :]
    ax[0].imshow(modality_np[slice_index, :, :], cmap="gray")
    ax[1].imshow(modality_np[slice_index, :, :], cmap="gray")
    ax[1].imshow(seg_slice, cmap="plasma", alpha=np.where(seg_slice > 0, 0.3, 0))

    for a in ax:
        a.axis("off")
    plt.tight_layout()


def visualize_inpainting(t1n_voided: Union[str, Path], prediction: Union[str, Path]):
    """Visualize the inpainting results

    Args:
        t1n_voided (Union[str, Path]): Voided T1 modality file
        prediction (Union[str, Path]): Inpainting prediction file
    """
    voided_np = nib.load(t1n_voided).get_fdata().transpose(2, 1, 0)
    inpainting_np = nib.load(prediction).get_fdata().transpose(2, 1, 0)
    _, ax = plt.subplots(1, 2, figsize=(8, 4))

    slice_index = voided_np.shape[0] // 2  # You can choose any slice here
    ax[0].imshow(voided_np[slice_index, :, :], cmap="gray")
    ax[1].imshow(inpainting_np[slice_index, :, :], cmap="gray")
    for a in ax:
        a.axis("off")
    plt.tight_layout()


def visualize_missing_mri_t2w(
    synthesized_t2w: Union[str, Path],
    data_folder: Union[str, Path] = DATA_FOLDER,
    subject_id: str = "BraTS-GLI-00001-000",
    slice_index: int = 75,
):
    """Visualize the MRI modalities for a given slice index

    Args:
        synthesized_t2w (Union[str, Path]): Path to the synthesized T2w file
        data_folder (Union[str, Path], optional): Path to the folder containing the t1, t1c, t2 & flair file. Defaults to DATA_FOLDER.
        subject_id (str, optional): Subject ID to visualize. Defaults to "BraTS-GLI-00001-000".
        slice_index (int, optional): Slice to be visualized (first index in data of shape (155, 240, 240)). Defaults to 75.
    """
    _, axes = plt.subplots(1, 5, figsize=(12, 10))

    subject_path = Path(data_folder) / subject_id
    modalities = ["t1n", "t1c", "t2f", "t2w"]
    for i, mod in enumerate(modalities):
        modality_file = subject_path / f"{subject_id}-{mod}.nii.gz"
        modality_np = nib.load(modality_file).get_fdata().transpose(2, 1, 0)
        axes[i].set_title(mod)
        axes[i].imshow(modality_np[slice_index, :, :], cmap="gray")
        axes[i].axis("off")

    # show synthetic T2w
    synthetic_t2w_np = nib.load(synthesized_t2w).get_fdata().transpose(2, 1, 0)
    axes[4].set_title("Synthesized t2w")
    axes[4].imshow(synthetic_t2w_np[slice_index, :, :], cmap="gray")
    axes[4].axis("off")
