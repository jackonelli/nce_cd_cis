import unittest
import torch

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit
from src.nce.cd_cnce import CdCnceCrit

from tests.nce.nce_test_utils import sample_postive_test_samples


class TestCdCnce(unittest.TestCase):
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
        print("HERE STARTS THE TEST")
        # Sample some data to test on
        num_samples = 5  # 1000
        y = torch.tensor([[-0.9742, -1.7647],
                          [-0.9475, -0.3340],
                          [-0.4262, 0.6364],
                          [1.9665, -1.4356],
                          [0.4696, 1.0241]])  # sample_postive_test_samples(num_samples)
        print("y")
        print(y)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
                (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()
        num_neg_samples = 3

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_true = torch.tensor([-0.2900, -0.7650])
        print("true mean")
        print(mu_true)
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise = torch.tensor([2.1008, -0.0943])
        print("noise mean")
        print(mu_noise)
        true_distr = GaussianModel(mu_true.clone(), cov_true.clone())
        noise_distr = ConditionalMultivariateNormal(cov_noise)

        mcmc_steps = 1
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)

        y = torch.repeat_interleave(y, num_neg_samples, dim=0)
        y_samples = torch.tensor([[[-1.9684, -1.3018]],

                                  [[-0.2309, -2.3210]],

                                  [[0.4499, -1.9452]],

                                  [[-2.5049, 0.5279]],

                                  [[-0.1199, -0.7333]],

                                  [[-1.8766, -0.3863]],

                                  [[-0.9815, -0.9619]],

                                  [[-0.3796, 0.4796]],

                                  [[-1.5267, 0.9659]],

                                  [[0.2503, -0.1551]],

                                  [[2.4362, -1.5338]],

                                  [[0.9104, 0.3883]],

                                  [[0.9689, -0.4269]],

                                  [[-0.6876, 0.5823]],

                                  [[-0.3902, 2.8169]]])  # criterion.sample_noise(1, y)

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
            print(grad / grad_ref)
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))

        print("HERE ENDS THE TEST")


    def test_several_steps(self):
        """Just check that everything seems to run for multiple MCMC steps"""

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
        criterion = CdCnceCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        criterion.calculate_crit_grad(y, None)


if __name__ == "__main__":
    unittest.main()
