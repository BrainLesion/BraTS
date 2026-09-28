import unittest

from brats.utils.cuda import normalize_cuda_devices


class TestNormalizeCudaDevices(unittest.TestCase):
    def test_single_device(self):
        self.assertEqual(normalize_cuda_devices("0"), "0")

    def test_multiple_devices(self):
        self.assertEqual(normalize_cuda_devices("0,1,2"), "0,1,2")

    def test_strips_whitespace_around_ids(self):
        self.assertEqual(normalize_cuda_devices(" 0 , 1 "), "0,1")

    def test_drops_empty_entries(self):
        self.assertEqual(normalize_cuda_devices("0,,1,"), "0,1")

    def test_whitespace_only_input_raises(self):
        with self.assertRaises(ValueError):
            normalize_cuda_devices(" , , ")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            normalize_cuda_devices("")
