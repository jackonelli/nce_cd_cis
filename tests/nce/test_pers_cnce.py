import unittest
import torch
import numpy as np

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.part_fn_base import norm_weights
from src.models.ebm.normal_params import NormalEbm


class TestPersistentCnce(unittest.TestCase):
    def test_criteria_equal_distr(self):
        pass


def y_weights_norm(y, y_samples, true_distr, noise_distr):
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (
        y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum()
    )

def test_crit(self):
    # Data dimension, num samples, num negative samples
    D, N, J = 3, 5, 10
    cov = torch.diag(torch.ones(D,))
    p_m = NormalEbm(torch.zeros(D,), cov )
    p_n = ConditionalMultivariateNormal(cov=cov)
    crit = PersistentCondNceCrit(p_m, p_n)
    y = torch.randn((N, D))
    y_samples = crit.sample_noise(J * y.size(0), y)
    loss = crit.crit(y, y_samples)

if __name__ == "__main__":
    unittest.main()
