import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.binary import NceBinaryCrit
from src.part_fn_base import unnorm_weights


class TestBinaryNCE(unittest.TestCase):
    def test_criteria_equal_distr(self):

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criteria = NceBinaryCrit(true_distr, noise_distr)

        y_samples = sample_negative_test_samples(criteria, y)
        res = criteria.crit(y, y_samples)

        # Reference calculation
        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0])
        ref = - torch.log(1 / (1 + num_neg_samples)) - num_neg_samples * torch.log(num_neg_samples / (1 + num_neg_samples))

        self.assertTrue(torch.allclose(ref, res))

    def test_criteria_example(self):

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criteria = NceBinaryCrit(true_distr, noise_distr)

        y_samples = sample_negative_test_samples(criteria, y)
        res = criteria.crit(y, y_samples)

        # Reference calculation
        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0])
        y_w = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ref_y = torch.log(y_w / (y_w + num_neg_samples)).mean()

        ys_w = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)
        ref_ys = num_neg_samples * torch.log(num_neg_samples / (ys_w + num_neg_samples)).mean()
        ref = - ref_y - ref_ys

        self.assertTrue(torch.allclose(ref, res))


def sample_postive_test_samples(num_samples, min_num_dims=2, max_num_dims=5):

    num_dims = np.random.randint(min_num_dims, max_num_dims)
    mu = torch.randn((num_dims,))
    y = torch.randn((num_samples, num_dims)) + mu

    return y


def sample_negative_test_samples(criteria, y, min_neg_samples=2, max_neg_samples=20):

    num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()
    y_samples = criteria.sample_noise(num_neg_samples * y.size(0), y)

    return y_samples


if __name__ == '__main__':
    unittest.main()