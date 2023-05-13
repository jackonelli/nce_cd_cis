import unittest
import torch

from src.aem.aem_pers_cond_smc import AemSmcCondPersCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemSmcCondPersCrit(unittest.TestCase):
    def test_crit(self):
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

        crit = AemSmcCondPersCrit(model, proposal, num_negative, batch_size=num_samples)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, num_samples)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

        # Check so that persistent samples were updated
        assert crit._persistent_y is not None

        num_updates = 10
        for i in range(num_updates):
            _, _, _ = crit.crit(y, num_samples)

        assert torch.allclose(crit.persistent_y(y, num_samples), crit._persistent_y)
        assert not torch.allclose(y, crit._persistent_y)

        _, _, _, y_samples, _ = crit.inner_pers_crit(y, y)
        assert torch.allclose(y_samples[:, 0, :], y)

if __name__ == "__main__":
    unittest.main()
