import unittest
import torch

from src.noise_distr.normal import MultivariateNormal
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit
from src.part_fn_base import unnorm_weights, cond_unnorm_weights

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestCondNce(unittest.TestCase):
    def test_criterion_equal_distr(self):
        """Check that criterion is correct for case model=noise distr."""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Set model and noise distr. to be equal
        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criterion = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        res = criterion.crit(y)

        # For model and noise_distr equal, criterion should be constant
        ref = torch.log(torch.tensor(2))

        self.assertTrue(torch.allclose(ref, res))

    def test_criterion_example_unc_distr(self):
        """Test example for calculating conditional NCE criterion when noise distr. is not conditional"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criterion = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise(num_neg_samples * y.size(0), y)
        w_tilde = criterion._unnorm_w(y, y_samples)
        res = criterion.inner_crit(y, y_samples)

        # Reference calculations (check so that weights are calculated and used as intended)
        y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ys_w_tilde = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)**(-1)

        w_tilde_ref = torch.cat([y_w_tilde[i] * ys_w_tilde[(num_neg_samples * i):(num_neg_samples * (i + 1))] for i
                                in range(num_samples)])
        ref = - torch.log(w_tilde_ref / (1 + w_tilde_ref)).mean()

        self.assertTrue(torch.allclose(w_tilde_ref, w_tilde))
        self.assertTrue(torch.allclose(ref, res))

    def test_criterion_example(self):
        """Test example for calculating conditional NCE criterion when noise distr. is not conditional"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        cov_noise = torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = ConditionalMultivariateNormal(cov_noise)
        criterion = CondNceCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise(num_neg_samples * y.size(0), y)
        w_tilde = criterion._unnorm_w(y, y_samples)
        res = criterion.inner_crit(y, y_samples)

        # Reference calculations (check so that weights are calculated and used as intended)
        w_tilde_ref = torch.cat([cond_unnorm_weights(y[i, :].reshape(1, -1),
                                                     y_samples[(num_neg_samples * i):(num_neg_samples * (i + 1)), :],
                                                     true_distr.prob, noise_distr.prob) for i in range(num_samples)])

        ref = - torch.log(w_tilde_ref / (1 + w_tilde_ref)).mean()

        self.assertTrue(torch.allclose(w_tilde_ref, w_tilde))
        self.assertTrue(torch.allclose(ref, res))


if __name__ == '__main__':
    unittest.main()
