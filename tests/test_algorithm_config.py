import unittest
from pathlib import Path

from brats.utils.algorithm_config import load_algorithms
from brats.utils.constants import META_DIR


class TestAlgorithmConfig(unittest.TestCase):

    def test_configs_valid(self):

        configs = [
            f
            for f in META_DIR.iterdir()
            if f.is_file() and f.suffix in [".yml", ".yaml"]
        ]
        print(configs)
        for config in configs:
            try:
                load_algorithms(file_path=config)
            except Exception as e:
                self.fail(f"Failed to load config {config}: {e}")
