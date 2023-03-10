import unittest

import numpy as np
import torch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.models.gaussian_model import GaussianModel


class TestPersistentCnceGrad(unittest.TestCase):
    def test_grad(self):
        """Check the evaluation of crit."""
        N, J, D = 10, 5, 3
        true_distr = GaussianModel(
            mu=torch.zeros(
                D,
            ),
            cov=torch.eye(D),
        )
        noise_distr = ConditionalMultivariateNormal(
            cov=3 * torch.eye(D),
        )

        torch.manual_seed(2)
        y = torch.randn((N, D)) + torch.randn((D,))

        # Make sure persistent y is updated
        idx = torch.arange(0, N)
        crit = PersistentCondNceCrit(true_distr, noise_distr, J)
        crit.calculate_crit_grad(y, idx)

        # Calculate grad without detach
        y = y.unsqueeze(dim=1).repeat(1, J, 1)
        y_p = crit.persistent_y(y, idx).reshape(-1, y.shape[-1])
        y_samples = crit.sample_noise(1, y_p)

        crit.calculate_inner_crit_grad(y_p, y_samples, y.reshape(-1, y.shape[-1]))
        grad_1 = [param.grad.detach().clone() for param in true_distr.parameters()]

        # Check so that gradients remain the same
        crit.calculate_inner_crit_grad(y_p, y_samples, y.reshape(-1, y.shape[-1]))
        grad_2 = [param.grad.detach().clone() for param in true_distr.parameters()]

        # Use detach on y_p
        crit.calculate_inner_crit_grad(y_p.detach().clone(), y_samples, y.reshape(-1, y.shape[-1]))
        grad_3 = [param.grad.detach().clone() for param in true_distr.parameters()]

        # Use detach on y_p an y_samples
        crit.calculate_inner_crit_grad(y_p.detach().clone(), y_samples.detach().clone(), y.reshape(-1, y.shape[-1]))
        grad_4 = [param.grad.detach().clone() for param in true_distr.parameters()]

        for g1, g2, g3, g4 in zip(grad_1, grad_2, grad_3, grad_4):
            assert torch.allclose(g1, g2)
            assert torch.allclose(g1, g3)
            assert torch.allclose(g1, g4)


if __name__ == "__main__":
    unittest.main()
