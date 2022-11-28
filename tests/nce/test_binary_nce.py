import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.binary import NceBinaryCrit
from src.part_fn_base import unnorm_weights


class TestBinaryNCE(unittest.TestCase):
    def test_criteria_equal_distr(self):

        num_dims = np.random.randint(2, 5)
        y_mu = torch.randn((num_dims,))
        num_samples = 1000
        y = torch.randn((num_samples, num_dims)) + y_mu

        mu, cov = torch.randn((num_dims,)), torch.eye(num_dims)
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)

        criteria = NceBinaryCrit(true_distr, noise_distr)

        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()
        y_samples = criteria.sample_noise(num_neg_samples * y.size(0), y)
        res = criteria.crit(y, y_samples)

        ref = - torch.log(1 / (1 + num_neg_samples)) - num_neg_samples * torch.log(num_neg_samples / (1 + num_neg_samples))
        self.assertTrue(torch.allclose(ref, res))

    def test_criteria_example(self):

        num_dims = np.random.randint(2, 5)
        y_mu = torch.randn((num_dims,))
        num_samples = 1000
        y = torch.randn((num_samples, num_dims)) + y_mu

        mu_true, cov_true = torch.randn((num_dims,)), torch.eye(num_dims)
        mu_noise, cov_noise = torch.randn((num_dims,)), torch.eye(num_dims)
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        criteria = NceBinaryCrit(true_distr, noise_distr)

        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()
        y_samples = criteria.sample_noise(num_neg_samples * y.size(0), y)
        res = criteria.crit(y, y_samples)

        y_w = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ref_y = torch.log(y_w / (y_w + num_neg_samples)).mean()

        ys_w = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)
        ref_ys = num_neg_samples * torch.log(num_neg_samples / (ys_w + num_neg_samples)).mean()
        ref = - ref_y - ref_ys

        self.assertTrue(torch.allclose(ref, res))


if __name__=='__main__':
    unittest.main()