import unittest
import torch
from src.part_fn_base import extend_sample, unnorm_weights, cond_unnorm_weights


class TestWeights(unittest.TestCase):
    def test_dims(self):
        y_sample = torch.randn(50, 2)
        n, d = y_sample.size()
        y = torch.randn(1, 2)
        y_all = extend_sample(y, y_sample)
        self.assertEqual(y_all.size(), (n + 1, d))
