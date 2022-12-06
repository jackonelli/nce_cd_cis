import unittest
import torch
from src.part_fn_utils import concat_samples, unnorm_weights, norm_weights


class TestWeights(unittest.TestCase):
    def test_dims(self):
        y_sample = torch.randn(1, 50, 2)
        n, j, d = y_sample.size()
        y = torch.randn(1, 2)
        y_all = concat_samples(y, y_sample)
        self.assertEqual(y_all.size(), (n, j + 1, d))

    def test_weight(self):
        y_sample = torch.ones((5,))
        w_tilde = unnorm_weights(y_sample, uniform, uniform)
        self.assertTrue(torch.allclose(torch.ones((5,)), w_tilde))

    def test_norm_weight(self):
        y_sample = torch.ones((5,))
        w_tilde = unnorm_weights(y_sample, uniform, uniform)
        w = norm_weights(w_tilde)
        self.assertTrue(torch.allclose(torch.ones((5,)) / 5, w))


def uniform(x):
    return torch.ones(x.size())
