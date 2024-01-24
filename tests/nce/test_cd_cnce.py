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
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true.clone(), cov_true.clone())
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        mcmc_steps = 1
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)

        y = torch.repeat_interleave(y, num_neg_samples, dim=0)
        y_samples = criterion.sample_noise(1, y)

        # Calculate gradient directly using CD+CNCE
        criterion.calculate_inner_crit_grad(y, y_samples)
        res = criterion.get_model_gradients()

        # Calculate gradient of CNCE crit.
        true_distr_ref = GaussianModel(mu_true.clone(), cov_true.clone())
        criterion_ref = CondNceCrit(true_distr_ref, noise_distr, num_neg_samples)
        criterion_ref.calculate_inner_crit_grad(y, y_samples)

        refs = criterion_ref.get_model_gradients()

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-3))

    def test_criterion_grad(self):
        """Check that criterion gives same gradient as NCE ranking for 1 step, when noise distr. is conditional"""
        print("TEST STARTS HERE")
        # Sample some data to test on
        num_samples = 5  # 1000
        y = torch.tensor([[-2.5278, 0.6083, 0.2062],
                          [-1.6150, 2.0330, -0.5338],
                          [-0.6505, 2.0204, 0.8260],
                          [-2.7263, 1.3239, 0.5762],
                          [-1.5593, 1.4321, 1.5634]])  # sample_postive_test_samples(num_samples)
        print("y")
        print(y)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true.clone(), cov_true.clone())
        noise_distr = ConditionalMultivariateNormal(cov_noise)

        mcmc_steps = 1
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)

        y = torch.repeat_interleave(y, num_neg_samples, dim=0)
        y_samples = torch.tensor([[[-3.5967e+00,  1.8068e+00,  3.9362e-01]],

        [[-3.3302e+00, -2.7416e-01, -3.9162e-02]],

        [[-8.2268e-02, -3.4417e-01, -2.0565e-01]],

        [[-1.4111e+00,  4.6796e-01, -6.5446e-01]],

        [[-2.7106e+00,  2.5717e+00, -4.7367e-01]],

        [[-1.4309e+00,  1.4161e+00, -1.6986e-01]],

        [[-2.8917e-01,  2.2081e+00,  2.5424e-01]],

        [[-1.3460e+00,  2.4110e-01, -1.1812e+00]],

        [[-4.3365e-03,  4.4114e+00, -1.6243e+00]],

        [[-2.1785e+00,  5.7454e-01,  1.5190e+00]],

        [[-3.9907e+00,  1.1558e+00,  4.9643e-01]],

        [[-2.6997e+00,  5.5572e-01,  1.1770e+00]],

        [[-1.9374e+00,  7.8639e-01,  1.2030e+00]],

        [[-2.0738e+00,  1.4319e+00, -1.4145e-01]],

        [[-8.6571e-01,  2.1864e+00,  1.5748e+00]]]) #criterion.sample_noise(1, y)

        print("y_samples")
        print(y_samples)

        # Calculate gradient directly using CD+CNCE
        criterion.calculate_inner_crit_grad(y, y_samples)
        res = criterion.get_model_gradients()

        # Calculate gradient of CNCE crit.
        true_distr_ref = GaussianModel(mu_true.clone(), cov_true.clone())
        criterion_ref = CondNceCrit(true_distr_ref, noise_distr, num_neg_samples)
        criterion_ref.calculate_inner_crit_grad(y, y_samples)
        refs = criterion_ref.get_model_gradients()

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))
        print("TEST ENDS HERE")

    def test_several_steps(self):
        """Just check that everything seems to run for multiple MCMC steps"""

        num_samples = 1000
        y = sample_postive_test_samples(num_samples)


        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()
        num_neg_samples = 3

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        cov_noise = torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = ConditionalMultivariateNormal(cov_noise)

        mcmc_steps = 3
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        criterion.calculate_crit_grad(y, None)


if __name__ == "__main__":
    unittest.main()
