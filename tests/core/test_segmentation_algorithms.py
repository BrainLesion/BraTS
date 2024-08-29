import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loguru import logger

from brats import (
    AdultGliomaSegmenter,
    AfricaSegmenter,
    MeningiomaSegmenter,
    MetastasesSegmenter,
    PediatricSegmenter,
)
from brats.utils.constants import (
    AdultGliomaAlgorithms,
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
        self.t1c = self.data_folder / "subject-t1c.nii.gz"
        self.t1n = self.data_folder / "subject-t1n.nii.gz"
        self.t2f = self.data_folder / "subject-t2f.nii.gz"
        self.t2w = self.data_folder / "subject-t2w.nii.gz"
        # Create dummy files
        for img in [self.t1c, self.t1n, self.t2f, self.t2w]:
            img.touch(exist_ok=True)

        self.segmenter = AdultGliomaSegmenter()

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

    def test_adult_glioma_segmenter_initialization(self):
        # Test default initialization
        segmenter = AdultGliomaSegmenter()
        self.assertIsInstance(segmenter, AdultGliomaSegmenter)

        # Test with custom arguments
        custom_segmenter = AdultGliomaSegmenter(
            algorithm=AdultGliomaAlgorithms.BraTS23_2, cuda_devices="1", force_cpu=True
        )
        self.assertIsInstance(custom_segmenter, AdultGliomaSegmenter)

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
