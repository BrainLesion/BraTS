import unittest
from pathlib import Path

from brats.algorithm_config import load_algorithms

PACKAGE_DIR = Path(__file__).parent.parent


class TestAlgorithmConfig(unittest.TestCase):

    def test_configs_valid(self):
        algorithms_folder = PACKAGE_DIR / "brats" / "algorithms"

        configs = algorithms_folder.iterdir()

        for config in configs:
            try:
                load_algorithms(file_path=config)
            except Exception as e:
                self.fail(f"Failed to load config {config}: {e}")
