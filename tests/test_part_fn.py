import unittest
import torch
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.models.gaussian_model import GaussianModel
from src.utils.part_fn_utils import (
    concat_samples,
    unnorm_weights,
    norm_weights,
    cond_unnorm_weights_ratio,
)


class TestWeights(unittest.TestCase):
    def test_concat_dims(self):
        y_sample = torch.randn(2, 5, 3)
        N, J, D = y_sample.size()
        y = torch.randn(N, D)
        y_all = concat_samples(y, y_sample)
        self.assertEqual(y_all.size(), (N, J + 1, D))

    def test_concat_order(self):
        N, J, D = 5, 50, 1
        y_sample = torch.ones(N, J, D)
        y = torch.zeros(N, D)
        y_all = concat_samples(y, y_sample)
        self.assertTrue(torch.allclose(y_all[:, 0, :], y))

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
