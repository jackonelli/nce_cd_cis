import unittest
import torch

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit
from src.nce.cd_cnce import CdCnceCrit

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestCdMH(unittest.TestCase):
    def test_several_steps(self):
        """Just check that everything seems to run"""

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        cov_noise = torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = ConditionalMultivariateNormal(cov_noise)

        mcmc_steps = 3
        criterion = CdCnceCrit(
            true_distr, noise_distr, num_neg_samples, mcmc_steps, save_metrics=False
        )
        criterion.calculate_crit_grad(y, None)


if __name__ == "__main__":
    unittest.main()
