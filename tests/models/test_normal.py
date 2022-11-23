import unittest
import torch

from src.models.ebm.normal_params import NormalEbm


class TestNormalEbm(unittest.TestCase):
    def test_prob(self):
        mu = torch.randn(1)
        sigma = torch.exp(torch.randn(1))

        sample = torch.randn(50)
        ref_distr = torch.distributions.Normal(mu, sigma)
        ref_prob = torch.exp(ref_distr.log_prob(sample))

        ebm = NormalEbm(mu, sigma**2)
        ebm_prob = ebm.prob(sample)

        self.assertTrue(torch.allclose(ref_prob, ebm_prob))

    def test_log_prob(self):
        mu = torch.randn(1)
        sigma = torch.exp(torch.randn(1))

        sample = torch.randn(50)
        ref_distr = torch.distributions.Normal(mu, sigma)
        ref_prob = torch.exp(ref_distr.log_prob(sample))

        ebm = NormalEbm(mu, sigma**2)
        ebm_prob = ebm.prob(sample)

        self.assertTrue(torch.allclose(ref_prob, ebm_prob))
