import unittest
import torch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.models.gaussian_model import GaussianModel


class TestPersistentCnce(unittest.TestCase):
    def test_crit(self):
        """Check the evaluation of crit."""
        N, J, D = 2, 5, 3
        true_distr = GaussianModel(
            mu=torch.zeros(
                D,
            ),
            cov=torch.eye(D),
        )
        noise_distr = ConditionalMultivariateNormal(
            cov=3 * torch.eye(D),
        )
        crit = PersistentCondNceCrit(true_distr, noise_distr, J)
        y = true_distr.mu.clone().repeat((N, 1))
        self.assertEqual(y.size(), (N, D))
        crit.calculate_crit_grad(
            y,
            torch.zeros(
                N,
            ),
        )

    def test_y_persistent_update(self):
        """Check that the persistent samples are properly updated

        Forces the weight of the actual sample to be zero and verifies that the persistent y
        is updated to a noisy sample from.
        """
        N, J, D = 2, 5, 3
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((N * J, 2), dtype=torch.long)
        # Zero weight for the actual sample.
        w_unnorm[0, 0] = 0.0
        y_samples = torch.ones((N * J, 1, D))
        y, idx = torch.ones((N * J, D)), torch.zeros((N,), dtype=torch.long)
        # Zero weight makes the updated persistent y be 1.
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        # This val. is needed in the API, but should not be selected.
        dummy_y = torch.randn(N, J, D)

        y_p = crit.persistent_y(dummy_y, idx)
        self.assertTrue(torch.allclose(y_p, torch.ones(dummy_y.size())))

    def test_y_persistent_dim(self):
        """Check that the persistent samples are properly updated

        Checks that the persistent sample has the same dim as actual y.
        """
        N, J, D = 2, 5, 1
        crit = PersistentCondNceCrit(None, None, J)
        log_w_unnorm = torch.ones((N * J, 2))
        y_samples = torch.ones((N * J, 1, D))
        y, idx = torch.ones((N * J, D)), torch.zeros((N,), dtype=torch.long)
        crit._update_persistent_y(torch.log(log_w_unnorm), y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(N, J, D), idx)

        self.assertEqual(y_p.size(), (N, J, D))

    def test_y_persistent_multi_dim(self):
        """Check that the persistent samples are properly updated

        Checks that the persistent sample has the same dim as actual y.
        For multi-dim y.
        """
        N, J, D = 1, 5, 2
        crit = PersistentCondNceCrit(None, None, J)
        w_unnorm = torch.ones((N * J, 2))
        y_samples = torch.ones((N * J, 1, D))
        y, idx = torch.ones((N * J, D)), torch.zeros((N,), dtype=torch.long)
        crit._update_persistent_y(w_unnorm, y, y_samples, idx)
        y_p = crit.persistent_y(torch.randn(N, J, D), idx)
        self.assertEqual(y_p.size(), (N, J, D))


if __name__ == "__main__":
    t = TestPersistentCnce()
    t.test_y_persistent_update()
    #unittest.main()
