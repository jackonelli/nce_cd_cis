import unittest
import torch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.part_fn_utils import norm_weights
from src.models.ebm.normal_params import NormalEbm


class TestPersistentCnce(unittest.TestCase):
    def test_y_persistent_update(self):
        N, J, D = 2, 5, 3
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((N, J + 1), dtype=torch.long)
        # Zero weight for the actual sample.
        w_unnorm[0, 0] = 0.0
        y_samples = torch.ones((N, J, D))
        y, idx = torch.ones((N, D)), torch.zeros((N,), dtype=torch.long)
        # Zero weight makes the updated persistent y be 1.
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        # This val. is needed in the API, but should not be selected.
        dummy_y = torch.randn(y.size())
        y_p = crit.persistent_y(dummy_y, idx)
        self.assertTrue(torch.allclose(y_p, torch.ones(y.size())))

    def test_y_persistent_dim(self):
        N, J, D = 1, 5, 1
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((N, J + 1))
        y_samples = torch.ones((N, J, D))
        y, idx = torch.ones((N, 1)), torch.zeros((N,), dtype=torch.long)
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(y.size()), idx)
        self.assertEqual(y_p.size(), y.size())

    def test_y_persistent_multi_dim(self):
        N, J, D = 1, 5, 2
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((N, J + 1))
        y_samples = torch.ones((N, J, D))
        y, idx = torch.ones((N, D)), torch.zeros((N,), dtype=torch.long)
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(y.size()), idx)
        self.assertEqual(y_p.size(), y.size())


if __name__ == "__main__":
    unittest.main()
