from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import Optional

from loguru import logger

from brats.core.brats_algorithm import BraTSAlgorithm
from brats.utils.constants import INPAINTING_ALGORITHMS, InpaintingAlgorithms, Task
from brats.utils.data_handling import input_sanity_check


class Inpainter(BraTSAlgorithm):

    def __init__(
        self,
        algorithm: InpaintingAlgorithms = InpaintingAlgorithms.BraTS23_1,
        cuda_devices: str | None = "0",
        force_cpu: bool = False,
    ):
        super().__init__(
            algorithm=algorithm,
            algorithms_file_path=INPAINTING_ALGORITHMS,
            task=Task.INPAINTING,
            cuda_devices=cuda_devices,
            force_cpu=force_cpu,
        )

    def _standardize_single_inputs(
        self, data_folder: Path, subject_id: str, inputs: dict[str, Path | str]
    ) -> None:
        """
        Standardize the input data to match the requirements of the selected algorithm.

        Args:
            data_folder (Path): Path to the data folder
            subject_id (str): Subject ID
            inputs (dict[str, Path | str]): Dictionary with the input data
        """

        subject_folder = data_folder / subject_id
        subject_folder.mkdir(parents=True, exist_ok=True)
        # TODO: investigate usage of symlinks (might cause issues on windows and would probably require different volume handling)
        t1n, mask = inputs["t1n"], inputs["mask"]
        try:
            shutil.copy(t1n, subject_folder / f"{subject_id}-t1n-voided.nii.gz")
            shutil.copy(mask, subject_folder / f"{subject_id}-mask.nii.gz")
        except FileNotFoundError as e:
            logger.error(f"Error while standardizing files: {e}")
            sys.exit(1)

        # sanity check inputs
        input_sanity_check(t1n=t1n, mask=mask)

    def infer_single(
        self,
        t1n: Path | str,
        mask: Path | str,
        output_file: Path | str,
        log_file: Optional[Path | str] = None,
    ) -> None:
        """Perform inpainting task on a single subject with the provided images and save the result to the output file.

        Args:
            t1n (Path | str): Path to the T1n image
            mask (Path | str): Path to the mask image
            output_file (Path | str): Path to save the segmentation
            log_file (Path | str, optional): Save logs to this file
        """
        self._infer_single(
            inputs={"t1n": t1n, "mask": mask},
            output_file=output_file,
            log_file=log_file,
        )
