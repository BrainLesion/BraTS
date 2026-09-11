import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from loguru import logger

from brats import Inpainter
from brats.constants import InpaintingAlgorithms

# --- Configuration for the full algorithm sweep (see test_all_inpainting_algorithms) ---
# Adjust these paths, or override via the BRATS_INPAINTING_T1N / BRATS_INPAINTING_MASK
# environment variables.
DEFAULT_T1N = (
    "/mnt/8tb_slot8/erikgro/test_set_2/"
    "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge_TestingDataset/"
    "BraTS-GLI-00010-001/BraTS-GLI-00010-001-t1n-voided.nii.gz"
)
DEFAULT_MASK = (
    "/mnt/8tb_slot8/erikgro/test_set_2/"
    "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge_TestingDataset/"
    "BraTS-GLI-00010-001/BraTS-GLI-00010-001-mask.nii.gz"
)
DEFAULT_OUTPUT_ROOT = Path("debug_outputs_inpainting")


class TestInpaintingAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_folder = self.test_dir / "data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.tmp_data_folder = self.test_dir / "tmp_std_data"
        self.tmp_data_folder.mkdir(parents=True, exist_ok=True)

        # Create mock paths for input images
        self.subject_folder = self.data_folder / "subject"
        self.subject_folder.mkdir(parents=True, exist_ok=True)
        self.t1n = self.subject_folder / "subject-t1n-voided.nii.gz"
        self.mask = self.subject_folder / "subject-mask.nii.gz"
        # Create dummy files
        for img in [self.t1n, self.mask]:
            img.touch(exist_ok=True)

        self.segmenter = Inpainter()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    # Standardization tests

    @patch("brats.core.inpainting_algorithms.input_sanity_check")
    def test_successful_single_standardization(self, mock_input_sanity_check):
        subject_id = "test_subject"
        self.segmenter._standardize_single_inputs(
            data_folder=self.tmp_data_folder,
            subject_id=subject_id,
            inputs={
                "t1n": self.t1n,
                "mask": self.mask,
            },
            subject_modality_separator="-",
        )
        subject_folder = self.tmp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        self.assertTrue((subject_folder / f"{subject_id}-t1n-voided.nii.gz").exists())
        self.assertTrue((subject_folder / f"{subject_id}-mask.nii.gz").exists())

    @patch("brats.core.inpainting_algorithms.input_sanity_check")
    @patch.object(logger, "error")
    def test_single_standardize_handle_file_not_found_error(
        self, mock_logger, mock_input_sanity_check
    ):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1n = "non_existent_file.nii.gz"
        with self.assertRaises(FileNotFoundError):
            self.segmenter._standardize_single_inputs(
                data_folder=self.data_folder,
                subject_id=subject_id,
                inputs={
                    "t1n": t1n,
                    "mask": self.mask,
                },
                subject_modality_separator="-",
            )
            mock_logger.assert_called()

    @patch("brats.core.inpainting_algorithms.Inpainter._standardize_single_inputs")
    def test_standardize_segmentation_inputs_list(self, mock_standardize_single_inputs):
        subjects = [f for f in self.data_folder.iterdir() if f.is_dir()]
        mapping = self.segmenter._standardize_batch_inputs(
            data_folder=self.tmp_data_folder,
            subjects=subjects,
            input_name_schema="BraTS-GLI-{id:05d}-000",
        )
        self.assertDictEqual(
            mapping,
            {
                "BraTS-GLI-00000-000": "subject",
            },
        )
        mock_standardize_single_inputs.assert_called_once()

    # Initialization tests

    def test_inpainter_initialization(self):
        # Test default initialization
        inpainter = Inpainter()
        self.assertIsInstance(inpainter, Inpainter)

        # Test with custom arguments
        custom_inpainter = Inpainter(
            algorithm=InpaintingAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_inpainter, Inpainter)

    # Integration sweep (requires docker, a GPU and real input data)

    @staticmethod
    def _selected_algorithms() -> list[InpaintingAlgorithms]:
        """Resolve which algorithms to run from BRATS_INPAINTING_ALGORITHMS."""
        subset = os.environ.get("BRATS_INPAINTING_ALGORITHMS", "").strip()
        if not subset:
            return list(InpaintingAlgorithms)
        wanted = {name.strip() for name in subset.split(",") if name.strip()}
        unknown = wanted - {a.value for a in InpaintingAlgorithms}
        if unknown:
            raise ValueError(f"Unknown algorithm(s): {sorted(unknown)}")
        return [a for a in InpaintingAlgorithms if a.value in wanted]

    def _check_output(
        self, output_dir: Path, output_file: Path, t1n: str, mask: str
    ) -> None:
        """Assert the algorithm produced exactly one usable output volume."""
        # exactly one file, and it is the one we asked for
        produced = sorted(output_dir.iterdir())
        self.assertEqual(
            [output_file],
            produced,
            f"Expected exactly one output file, got: {[p.name for p in produced]}",
        )
        self.assertGreater(output_file.stat().st_size, 0, "Output file is empty")

        # loadable, and geometrically consistent with the input
        inference = nib.load(output_file)
        voided = nib.load(t1n)
        self.assertEqual(
            voided.shape, inference.shape, "Output shape differs from input shape"
        )
        self.assertTrue(
            np.allclose(voided.affine, inference.affine, atol=1e-4),
            "Output affine differs from input affine",
        )

        # the volume is not blank, and the voided region was actually filled
        data = inference.get_fdata()
        self.assertGreater(np.count_nonzero(data), 0, "Output contains only zeros")
        mask_data = nib.load(mask).get_fdata() > 0
        self.assertGreater(
            np.count_nonzero(data[mask_data]),
            0,
            "Output is all zeros inside the mask (nothing was inpainted)",
        )

    @unittest.skipUnless(
        os.environ.get("BRATS_RUN_ALL_INPAINTING"),
        "Set BRATS_RUN_ALL_INPAINTING=1 to run the full algorithm sweep "
        "(pulls many GB of images, can take hours).",
    )
    def test_all_inpainting_algorithms(self):
        """Run every inpainting algorithm on one subject and check its output.

        Each algorithm gets its own output directory so that "exactly one output
        file" can be asserted. Failures are collected per algorithm via subTest,
        so one broken image does not hide the state of the others. A summary
        table is printed at the end.
        """
        t1n = os.environ.get("BRATS_INPAINTING_T1N", DEFAULT_T1N)
        mask = os.environ.get("BRATS_INPAINTING_MASK", DEFAULT_MASK)
        cuda_devices = os.environ.get("BRATS_CUDA_DEVICES", "0")
        output_root = Path(os.environ.get("BRATS_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT))

        for path in (t1n, mask):
            if not Path(path).is_file():
                self.skipTest(f"Input file not found: {path}")

        algorithms = self._selected_algorithms()
        log_dir = output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        results: list[tuple[str, str, float, str]] = []

        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm.value):
                # fresh, empty output dir per algorithm
                output_dir = output_root / algorithm.value
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True)
                output_file = output_dir / f"{algorithm.value}.nii.gz"

                print(f"\n=== {algorithm.value} ===", flush=True)
                start = time.time()
                try:
                    inpainter = Inpainter(
                        algorithm=algorithm, cuda_devices=cuda_devices
                    )
                    inpainter.infer_single(
                        t1n=t1n,
                        mask=mask,
                        output_file=output_file,
                        log_file=log_dir / f"{algorithm.value}.log",
                    )
                    self._check_output(
                        output_dir=output_dir,
                        output_file=output_file,
                        t1n=t1n,
                        mask=mask,
                    )
                except Exception as exc:
                    reason = str(exc).strip().splitlines()
                    results.append(
                        (
                            algorithm.value,
                            "FAIL",
                            time.time() - start,
                            reason[0][:120] if reason else type(exc).__name__,
                        )
                    )
                    raise
                results.append((algorithm.value, "OK", time.time() - start, ""))

        # Summary (printed even when individual subTests failed)
        print("\n" + "=" * 78)
        print(f"{'algorithm':<14}{'status':<8}{'runtime':>10}  detail")
        print("-" * 78)
        for name, status, elapsed, detail in results:
            print(f"{name:<14}{status:<8}{elapsed:>9.1f}s  {detail}")
        n_ok = sum(1 for r in results if r[1] == "OK")
        print("-" * 78)
        print(f"{n_ok}/{len(results)} algorithms produced a valid output")
        print(f"outputs: {output_root.resolve()}")
        print("=" * 78, flush=True)
