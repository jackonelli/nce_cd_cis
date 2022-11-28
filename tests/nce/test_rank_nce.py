import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.part_fn_base import unnorm_weights

from tests.nce.test_binary_nce import sample_postive_test_samples, sample_negative_test_samples


class TestRankNCE(unittest.TestCase):
    def test_criteria_equal_distr(self):

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criteria = NceRankCrit(true_distr, noise_distr)

        y_samples = sample_negative_test_samples(criteria, y)
        res = criteria.crit(y, y_samples)

        # Reference calculation
        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0])
        ref = torch.log(num_neg_samples + 1)

        self.assertTrue(torch.allclose(ref, res))

    def test_criteria_example(self):

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criteria = NceRankCrit(true_distr, noise_distr)

        y_samples = sample_negative_test_samples(criteria, y)
        res = criteria.crit(y, y_samples)

        # Reference calculation
        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0]).int()
        y_w = torch.tensor([y_weights_norm(y[i, :], y_samples[(num_neg_samples * i):(num_neg_samples * (i + 1)), :],
                                           true_distr, noise_distr) for i in range(num_samples)])
        ref = - torch.log(y_w).mean()

        self.assertTrue(torch.allclose(ref, res))


def y_weights_norm(y, y_samples, true_distr, noise_distr):
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum())


if __name__ == '__main__':
    unittest.main()