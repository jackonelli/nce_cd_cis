import unittest
import torch

from src.aem.aem_cis_joint_z_adapt import AemCisJointAdaCrit
from src.aem.aem_cis_joint_z_adapt_alt import AemCisJointAdaAltCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemCisAdaAltCrit(unittest.TestCase):

    def test_grad(self):
        # Check so that gradient is same as other AEM CIS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

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

        crit = AemCisJointAdaAltCrit(model, proposal, num_negative)

        # Calculate grads
        loss, _, _, y_samples, _ = crit.inner_pers_crit(y, y)
        loss.backward()
        grad_model = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal = [param.grad.detach().clone() for param in made.parameters()]

        proposal_ref = MockProposal(made, num_context_units, num_components, y_samples)
        crit_ref = AemCisJointAdaCrit(model, proposal_ref, num_negative)

        # Calculate ref. grads.
        model.clear_gradients()
        made.clear_gradients()
        loss_ref, _, _, y_samples_ref = crit_ref.inner_crit(y)
        loss_ref.backward()
        grad_model_ref = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal_ref = [param.grad.detach().clone() for param in made.parameters()]

        assert torch.allclose(y_samples[y.shape[0]:, :], y_samples_ref)

        assert y_samples.shape == (num_samples * (num_negative + 1), num_features)

        # Check grads same
        for p, p_ref in zip(grad_model, grad_model_ref):
            assert torch.abs(p-p_ref).max() < 1e-5
            assert torch.allclose(p, p_ref, rtol=1e-3, atol=1e-5)

        for p, p_ref in zip(grad_proposal, grad_proposal_ref):
            assert torch.abs(p-p_ref).max() < 1e-5
            assert torch.allclose(p, p_ref, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
