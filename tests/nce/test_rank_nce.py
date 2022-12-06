import unittest
import torch
import numpy as np

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.part_fn_utils import unnorm_weights

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestRankNCE(unittest.TestCase):
    def test_criterion_equal_distr(self):
        """Check that criterion is correct for case model=noise distr."""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Set model and noise distr. to be equal
        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criterion = NceRankCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        res = criterion.crit(y, None)

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
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criterion = NceRankCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise((y.size(0), num_neg_samples), y)
        res = criterion.inner_crit(y, y_samples, None)

        # Reference calculation (check so that positive and negative samples are used correctly)
        w_tilde_y = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        w_tilde_y_samples = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)
        y_w = torch.tensor([w_tilde_y[i] / (w_tilde_y[i] + w_tilde_y_samples[i, :].sum())
                            for i in range(num_samples)])
        ref = -torch.log(y_w).mean()

        self.assertTrue(torch.allclose(ref, res))


if __name__ == "__main__":
    unittest.main()
