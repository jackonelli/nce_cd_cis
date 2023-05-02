import unittest
import torch

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemCisCrit(unittest.TestCase):
    def test_crit(self):
        # Sanity check comparing with IS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((1,)).repeat(num_samples, 1)

        class MockProposal(AemJointProposal):
            """Mock proposal that always samples y"""

            def __init__(self, autoregressive_net, num_context_units, num_components, y, num_negative):
                super().__init__(autoregressive_net, num_context_units, num_components)
                self._y = y
                self._num_neg = num_negative
                self.counter = 0

            def inner_sample(self, distr, num_samples):
                # Always return y (weights will be the same for all obs.)

                if self.counter >= y.shape[-1]:
                    self.counter = 0

                self.counter += 1
                return self._y[:, self.counter - 1].reshape(-1, 1, 1).repeat(1, self._num_neg + 1, 1).reshape(-1, 1)

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = MockProposal(made, num_context_units, num_components, y, num_negative)

        crit = AemCisJointCrit(model, proposal, num_negative)
        crit_ref = AemIsJointCrit(model, proposal, num_negative)

        _, loss_p, loss_q, y_samples = crit.inner_crit(y)

        assert y_samples.grad is None

        _, loss_p_ref, loss_q_ref, y_samples_ref = crit_ref.inner_crit(y)

        y_s = y_samples.reshape(-1, num_negative, num_features)
        assert torch.allclose(y_s[:, 0, :], y)
        assert torch.allclose(y_samples, y_samples_ref)
        assert torch.allclose(loss_q, loss_q_ref)
        assert torch.allclose(loss_p, loss_p_ref)

        assert y_samples.shape == (num_samples * num_negative, num_features)

    def test_log_prob(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemCisJointCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        log_prob_p, log_prob_q = crit.log_prob(y)

        assert log_prob_p.shape == (num_samples,)
        assert log_prob_q.shape == (num_samples,)

        mean_log_prob_p, mean_log_prob_q = log_prob_p.mean(), log_prob_q.mean()
        assert not torch.isnan(mean_log_prob_p) or torch.isinf(mean_log_prob_p)
        assert not torch.isnan(mean_log_prob_q) or torch.isinf(mean_log_prob_q)
        assert log_prob_p.std() > 0.0
        assert log_prob_q.std() > 0.0


if __name__ == "__main__":
    unittest.main()
