import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.part_fn_base import unnorm_weights


class TestRankNCE(unittest.TestCase):
    def test_criteria_equal_distr(self):

        num_dims = np.random.randint(2, 5)
        y_mu = torch.randn((num_dims,))
        num_samples = 1000
        y = torch.randn((num_samples, num_dims)) + y_mu

        mu, cov = torch.randn((num_dims,)), torch.eye(num_dims)
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)

        criteria = NceRankCrit(true_distr, noise_distr)

        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()
        y_samples = criteria.sample_noise(num_neg_samples * y.size(0), y)
        res = criteria.crit(y, y_samples)

        ref = torch.log(num_neg_samples + 1)

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

        criteria = NceRankCrit(true_distr, noise_distr)

        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()
        y_samples = criteria.sample_noise(num_neg_samples * y.size(0), y)
        res = criteria.crit(y, y_samples)

        y_w = torch.tensor([y_weights_norm(y[i, :], y_samples[(num_neg_samples * i):(num_neg_samples * (i + 1)), :],
                                           true_distr, noise_distr) for i in range(num_samples)])
        ref = - torch.log(y_w).mean()

        print(ref)
        print(res)

        self.assertTrue(torch.allclose(ref, res))


def y_weights_norm(y, y_samples, true_distr, noise_distr):
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum())


if __name__=='__main__':
    unittest.main()