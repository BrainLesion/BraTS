import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from brats import (
    AdultGliomaPostTreatmentSegmenter,
    AdultGliomaPreTreatmentSegmenter,
    AfricaSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
    PediatricSegmenter,
)
from brats.constants import (
    AdultGliomaPostTreatmentAlgorithms,
    AdultGliomaPreTreatmentAlgorithms,
    AfricaAlgorithms,
    MeningiomaAlgorithms,
    MetastasesAlgorithms,
    PediatricAlgorithms,
)


class TestSegmentationAlgorithms(unittest.TestCase):
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
        self.t1c = self.subject_folder / "subject-t1c.nii.gz"
        self.t1n = self.subject_folder / "subject-t1n.nii.gz"
        self.t2f = self.subject_folder / "subject-t2f.nii.gz"
        self.t2w = self.subject_folder / "subject-t2w.nii.gz"
        # Create dummy files
        for img in [self.t1c, self.t1n, self.t2f, self.t2w]:
            img.touch(exist_ok=True)

        self.segmenter = AdultGliomaPostTreatmentSegmenter()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

    ### Standardization tests

    @patch("brats.core.segmentation_algorithms.input_sanity_check")
    def test_successful_single_standardization(self, mock_input_sanity_check):
        subject_id = "test_subject"
        self.segmenter._standardize_single_inputs(
            data_folder=self.tmp_data_folder,
            subject_id=subject_id,
            inputs={
                "t1c": self.t1c,
                "t1n": self.t1n,
                "t2f": self.t2f,
                "t2w": self.t2w,
            },
            subject_modality_separator="-",
        )
        subject_folder = self.tmp_data_folder / subject_id
        self.assertTrue(subject_folder.exists())
        for img_type in ["t1c", "t1n", "t2f", "t2w"]:
            self.assertTrue(
                (subject_folder / f"{subject_id}-{img_type}.nii.gz").exists()
            )

    @patch("brats.core.segmentation_algorithms.input_sanity_check")
    @patch("sys.exit")
    @patch.object(logger, "error")
    def test_single_standardize_handle_file_not_found_error(
        self, mock_logger, mock_exit, mock_input_sanity_check
    ):
        subject_id = "test_subject"
        # Provide a non-existent file path for t1c
        t1c = "non_existent_file.nii.gz"
        self.segmenter._standardize_single_inputs(
            data_folder=self.data_folder,
            subject_id=subject_id,
            inputs={
                "t1c": t1c,
                "t1n": self.t1n,
                "t2f": self.t2f,
                "t2w": self.t2w,
            },
            subject_modality_separator="-",
        )
        mock_logger.assert_called()
        mock_exit.assert_called_with(1)

    @patch(
        "brats.core.segmentation_algorithms.SegmentationAlgorithm._standardize_single_inputs"
    )
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

    ### Initialization tests

    def test_adult_glioma_pre_op_segmenter_initialization(self):
        # Test default initialization
        segmenter = AdultGliomaPreTreatmentSegmenter()
        self.assertIsInstance(segmenter, AdultGliomaPreTreatmentSegmenter)

        # Test with custom arguments
        custom_segmenter = AdultGliomaPreTreatmentSegmenter(
            algorithm=AdultGliomaPreTreatmentAlgorithms.BraTS23_2,
            cuda_devices="1",
            force_cpu=True,
        )
        self.assertIsInstance(custom_segmenter, AdultGliomaPreTreatmentSegmenter)

    def test_adult_glioma_post_op_segmenter_initialization(self):
        # Test default initialization
        segmenter = AdultGliomaPostTreatmentSegmenter()
        self.assertIsInstance(segmenter, AdultGliomaPostTreatmentSegmenter)

        # Test with custom arguments
        custom_segmenter = AdultGliomaPostTreatmentSegmenter(
            algorithm=AdultGliomaPostTreatmentAlgorithms.BraTS24_2,
            cuda_devices="1",
            force_cpu=True,
        )
        self.assertIsInstance(custom_segmenter, AdultGliomaPostTreatmentSegmenter)

    def test_meningioma_segmenter_initialization(self):
        # Test default initialization
        segmenter = MeningiomaSegmenter()
        self.assertIsInstance(segmenter, MeningiomaSegmenter)

        # Test with custom arguments
        custom_segmenter = MeningiomaSegmenter(
            algorithm=MeningiomaAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_segmenter, MeningiomaSegmenter)

    def test_pediatric_segmenter_initialization(self):
        # Test default initialization
        segmenter = PediatricSegmenter()
        self.assertIsInstance(segmenter, PediatricSegmenter)

        # Test with custom arguments
        custom_segmenter = PediatricSegmenter(
            algorithm=PediatricAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_segmenter, PediatricSegmenter)

    def test_africa_segmenter_initialization(self):
        # Test default initialization
        segmenter = AfricaSegmenter()
        self.assertIsInstance(segmenter, AfricaSegmenter)

        # Test with custom arguments
        custom_segmenter = AfricaSegmenter(
            algorithm=AfricaAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_segmenter, AfricaSegmenter)

    def test_metastases_segmenter_initialization(self):
        # Test default initialization
        segmenter = MetastasesSegmenter()
        self.assertIsInstance(segmenter, MetastasesSegmenter)

        # Test with custom arguments
        custom_segmenter = MetastasesSegmenter(
            algorithm=MetastasesAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_segmenter, MetastasesSegmenter)

    ## Test MeningiomaSegmenter specialty

    @patch("brats.core.segmentation_algorithms.MeningiomaSegmenter._infer_single")
    def test_meningioma_segmenter_infer_single_2023_valid(self, mock_infer_single):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS23_1)

        segmenter.infer_single(
            t1c=self.t1c,
            t1n=self.t1n,
            t2f=self.t2f,
            t2w=self.t2w,
            output_file=self.tmp_data_folder / "output.nii.gz",
        )

        mock_infer_single.assert_called_once()

    @patch("brats.core.segmentation_algorithms.MeningiomaSegmenter._infer_single")
    def test_meningioma_segmenter_infer_single_2023_invalid_missing_modalities(
        self, mock_infer_single
    ):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS23_1)

        with self.assertRaises(ValueError):
            segmenter.infer_single(
                t1c=self.t1c,
                # t1n=self.t1n,  # Missing modality
                t2f=self.t2f,
                t2w=self.t2w,
                output_file=self.tmp_data_folder / "output.nii.gz",
            )

        mock_infer_single.assert_not_called()

    @patch("brats.core.segmentation_algorithms.MeningiomaSegmenter._infer_single")
    def test_meningioma_segmenter_infer_single_2024_valid(self, mock_infer_single):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS24_1)

        segmenter.infer_single(
            t1c=self.t1c,
            output_file=self.tmp_data_folder / "output.nii.gz",
        )

        mock_infer_single.assert_called_once()

    @patch("brats.core.segmentation_algorithms.MeningiomaSegmenter._infer_single")
    def test_meningioma_segmenter_infer_single_2024_invalid_missing_t1c(
        self, mock_infer_single
    ):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS24_1)

        with self.assertRaises(ValueError):
            segmenter.infer_single(
                # t1c=self.t1c,
                # t1n=self.t1n,
                # t2f=self.t2f,
                t2w=self.t2w,
                output_file=self.tmp_data_folder / "output.nii.gz",
            )

        mock_infer_single.assert_not_called()

    @patch("brats.core.segmentation_algorithms.MeningiomaSegmenter._infer_single")
    def test_meningioma_segmenter_infer_single_2024_invalid_too_many_files(
        self, mock_infer_single
    ):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS24_1)

        with self.assertRaises(ValueError):
            segmenter.infer_single(
                t1c=self.t1c,
                # t1n=self.t1n,
                # t2f=self.t2f,
                t2w=self.t2w,
                output_file=self.tmp_data_folder / "output.nii.gz",
            )

        mock_infer_single.assert_not_called()

    @patch("brats.core.brats_algorithm.BraTSAlgorithm._process_batch_output")
    @patch("brats.core.brats_algorithm.run_container")
    @patch(
        "brats.core.segmentation_algorithms.MeningiomaSegmenter._standardize_single_inputs"
    )
    def test_meningioma_segmenter_infer_batch_2024(
        self,
        mock_standardize_single_inputs,
        mock_run_container,
        mock_process_batch_output,
    ):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS24_1)

        segmenter.infer_batch(
            data_folder=self.data_folder,
            output_folder=self.tmp_data_folder,
            log_file=self.tmp_data_folder / "log.txt",
        )

        self.assertEqual(mock_standardize_single_inputs.call_count, 1)
        # assert algorithms was called with just t1c
        args, kwargs = mock_standardize_single_inputs.call_args
        input_keys = list(kwargs["inputs"].keys())
        self.assertEqual(input_keys, ["t1c"])

        self.assertEqual(mock_run_container.call_count, 1)
        self.assertEqual(mock_process_batch_output.call_count, 1)

    @patch("brats.core.brats_algorithm.BraTSAlgorithm._process_batch_output")
    @patch("brats.core.brats_algorithm.run_container")
    @patch(
        "brats.core.segmentation_algorithms.MeningiomaSegmenter._standardize_single_inputs"
    )
    def test_meningioma_segmenter_infer_batch_2023(
        self,
        mock_standardize_single_inputs,
        mock_run_container,
        mock_process_batch_output,
    ):
        segmenter = MeningiomaSegmenter(algorithm=MeningiomaAlgorithms.BraTS23_1)

        segmenter.infer_batch(
            data_folder=self.data_folder,
            output_folder=self.tmp_data_folder,
            log_file=self.tmp_data_folder / "log.txt",
        )

        self.assertEqual(mock_standardize_single_inputs.call_count, 1)
        # assert algorithms was called with all modalities
        args, kwargs = mock_standardize_single_inputs.call_args
        input_keys = list(kwargs["inputs"].keys())
        self.assertEqual(input_keys, ["t1c", "t1n", "t2f", "t2w"])

        self.assertEqual(mock_run_container.call_count, 1)
        self.assertEqual(mock_process_batch_output.call_count, 1)
