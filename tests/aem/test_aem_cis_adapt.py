import unittest
import torch

from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.aem.aem_cis_joint_z_adapt import AemCisJointAdaCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemCisAdaCrit(unittest.TestCase):
    def test_crit(self):
        # Sanity check comparing with standard CIS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        class MockProposal(AemJointProposal):
            """Mock proposal that always samples y_samples"""

            def __init__(self, autoregressive_net, num_context_units, num_components, y_samples):
                super().__init__(autoregressive_net, num_context_units, num_components)
                self._y = y_samples
                self._num_neg = num_negative
                self.counter = 0

            def inner_sample(self, distr, num_samples):
                # Always return y (weights will be the same for all obs.)

                if self.counter >= y.shape[-1]:
                    self.counter = 0

                self.counter += 1

                return self._y[:, self.counter - 1]

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemCisJointAdaCrit(model, proposal, num_negative)
        _, loss_p, q_loss, y_samples = crit.inner_crit(y)

        proposal_ref = MockProposal(made, num_context_units, num_components, torch.cat((y, y_samples)))
        crit_ref = AemCisJointCrit(model, proposal_ref, num_negative)
        _, loss_p_ref, _, y_samples_ref = crit_ref.inner_crit(y)

        assert torch.allclose(y_samples, y_samples_ref)
        assert torch.allclose(loss_p, loss_p_ref)

        assert q_loss.shape == torch.Size([])
        assert not torch.isnan(q_loss) or torch.isinf(q_loss)


if __name__ == "__main__":
    unittest.main()
