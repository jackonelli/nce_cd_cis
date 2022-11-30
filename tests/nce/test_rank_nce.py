import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.part_fn_base import unnorm_weights

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestRankNCE(unittest.TestCase):
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
        criterion = NceRankCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        res = criterion.crit(y)

        # For model and noise_distr equal, criteria should depend only on the number of neg. samples
        ref = torch.log(num_neg_samples + 1)

        self.assertTrue(torch.allclose(ref, res))

    def test_criterion_example(self):
        """Test example for calculating NCE ranking criterion"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = MultivariateNormal(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criterion = NceRankCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise(num_neg_samples * y.size(0), y)
        res = criterion.inner_crit(y, y_samples)

        # Reference calculation (check so that positive and negative samples are used correctly)
        y_w = torch.tensor(
            [
                norm_weights_y(
                    y[i, :],
                    y_samples[(num_neg_samples * i) : (num_neg_samples * (i + 1)), :],
                    true_distr,
                    noise_distr,
                )
                for i in range(num_samples)
            ]
        )
        ref = -torch.log(y_w).mean()

        self.assertTrue(torch.allclose(ref, res))


def norm_weights_y(y, y_samples, true_distr, noise_distr):
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j)"""
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (
        y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum()
    )


if __name__ == "__main__":
    unittest.main()
