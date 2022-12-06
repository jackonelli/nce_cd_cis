import unittest
import torch

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit
from src.nce.cd_cnce import CdCnceCrit

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestCdRank(unittest.TestCase):

    def test_criterion_grad_unc_distr(self):
        """Check that criterion gives same gradient as NCE ranking for 1 step, when noise distr. is not conditional"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        mcmc_steps = 1
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)

        y = torch.repeat_interleave(y, num_neg_samples, dim=0)
        y_samples = criterion.sample_noise((y.size(0), 1), y)

        # Calculate gradient directly using CD+CNCE
        criterion.calculate_inner_crit_grad(y, y_samples, None)
        res = criterion.get_model_gradients()

        # Calculate gradient of CNCE crit.
        true_distr_ref = GaussianModel(mu_true, cov_true)
        criterion_ref = CondNceCrit(true_distr_ref, noise_distr, num_neg_samples)
        criterion_ref.calculate_inner_crit_grad(y, y_samples, None)

        refs = criterion_ref.get_model_gradients()

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))

    def test_several_steps(self):
        """Just check that everything seems to run for multiple MCMC steps"""

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        mcmc_steps = 3
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        criterion.calculate_crit_grad(y, None)




if __name__ == '__main__':
    unittest.main()
