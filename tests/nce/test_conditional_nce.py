import unittest
import torch

from src.noise_distr.normal import MultivariateNormal
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit
from src.part_fn_base import unnorm_weights, cond_unnorm_weights

from tests.nce.test_binary_nce import sample_postive_test_samples, sample_negative_test_samples


class TestCondNce(unittest.TestCase):
    def test_criteria_equal_distr(self):

        num_samples = 1000
        num_neg_samples = 5
        y = sample_postive_test_samples(num_samples)

        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criteria = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        res = criteria.crit(y)

        # Reference calculation
        ref = torch.log(torch.tensor(2))

        self.assertTrue(torch.allclose(ref, res))

    def test_criteria_unc_distr(self):

        num_samples = 1000
        num_neg_samples = 5
        y = sample_postive_test_samples(num_samples)

        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criteria = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        y_samples = sample_negative_test_samples(criteria, y)
        w_tilde = criteria._unnorm_w(y, y_samples)
        res = criteria.crit(y)

        # Reference calculations
        y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ys_w_tilde = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)**(-1)

        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0]).int()
        w_tilde_ref = torch.cat([y_w_tilde[i] * ys_w_tilde[(num_neg_samples * i):(num_neg_samples * (i + 1))] for i
                                in range(num_samples)])
        ref = - torch.log(w_tilde_ref / (1 + w_tilde_ref)).mean()

        self.assertTrue(torch.allclose(w_tilde_ref, w_tilde))
        self.assertTrue(torch.allclose(ref, res))

    def test_criteria_example(self):

        num_samples = 1000
        num_neg_samples = 5
        y = sample_postive_test_samples(num_samples)

        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = ConditionalMultivariateNormal(cov_noise)
        criteria = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        y_samples = sample_negative_test_samples(criteria, y)
        w_tilde = criteria._unnorm_w(y, y_samples)
        res = criteria.crit(y)

        num_neg_samples = torch.tensor(y_samples.shape[0] / y.shape[0]).int()
        w_tilde_ref = torch.cat([cond_unnorm_weights(y[i, :].reshape(1, -1),
                                                        y_samples[(num_neg_samples * i):(num_neg_samples * (i + 1)), :],
                                                        true_distr.prob, noise_distr.prob) for i in range(num_samples)])

        self.assertTrue(torch.allclose(w_tilde_ref, w_tilde))

        ref = - torch.log(w_tilde_ref / (1 + w_tilde_ref)).mean()

        self.assertTrue(torch.allclose(ref, res))


if __name__ == '__main__':
    unittest.main()
