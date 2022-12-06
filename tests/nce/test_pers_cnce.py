import unittest
import torch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.part_fn_base import norm_weights
from src.models.ebm.normal_params import NormalEbm


class TestPersistentCnce(unittest.TestCase):
    def test_y_persistent_update(self):
        J = 5
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((J + 1,))
        y_samples = torch.ones((J + 1,))
        y, idx = torch.ones((1,)), torch.zeros((1,))
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(y.size()), idx)
        self.assertAlmostEqual(y_p.item(), 1)

    def test_y_persistent_dim(self):
        J, D = 5, 10
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((J + 1,))
        y_samples = torch.ones((J + 1, D))
        y, idx = torch.ones((1,)), torch.zeros((1,))
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(y.size()), idx)
        self.assertEqual(y_p.size(), y.size())

    def test_y_persistent_multi_dim(self):
        J = 5
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((J + 1,))
        y_samples = torch.ones((J + 1,))
        y, idx = torch.ones((1,)), torch.zeros((1,))
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(y.size()), idx)
        self.assertAlmostEqual(y_p.item(), 1)


if __name__ == "__main__":
    unittest.main()
