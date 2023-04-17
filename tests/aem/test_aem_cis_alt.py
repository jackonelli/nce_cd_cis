import unittest
import torch

from src.aem.aem_cis_alt import AceCisJointAltCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.models.aem.made import ResidualMADE
from src.models.aem.energy_net import ResidualEnergyNet
from src.noise_distr.aem_proposal import AemProposal


class TestAemCisAlt(unittest.TestCase):

    def test_grad(self):
        # Check so that gradient is same as other AEM CIS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()
        num_mixture_components = 10

        output_dim_multiplier = num_context_units + 3 * num_mixture_components

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))

        made = ResidualMADE(input_dim=num_features, n_residual_blocks=2, hidden_dim=10,
                            output_dim_multiplier=output_dim_multiplier)

        proposal = AemProposal(autoregressive_net=made, proposal_component_family='gaussian',
                               num_context_units=num_context_units, num_components=num_mixture_components)

        crit = AceCisJointAltCrit(model, proposal, num_negative)
        crit_ref = AemCisJointCrit(model, proposal, num_negative)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        q, context = proposal.forward(y)
        y_samples = crit.inner_sample_noise(q, num_samples=num_negative)

        # Calculate grads.
        loss, _, _, _ = crit.inner_crit((y, context), (y_samples, q))
        loss.backward(retain_graph=True)
        grad_model = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal = [param.grad.detach().clone() for param in proposal.parameters()] # made.parameters()

        # Calculate ref. grads.
        model.clear_gradients()
        made.clear_gradients()
        loss_ref, _, _ = crit_ref.inner_crit((y, context), (y_samples, q))
        loss_ref.backward()
        grad_model_ref = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal_ref = [param.grad.detach().clone() for param in proposal.parameters()]

        # Check parameter grads same
        for p, p_ref in zip(grad_model, grad_model_ref):
            assert torch.max(torch.abs(p-p_ref)) < 1e-6
            assert torch.allclose(p, p_ref, atol=1e-6, rtol=1e-3)

        for p, p_ref in zip(grad_proposal, grad_proposal_ref):
            assert torch.allclose(p, p_ref)


if __name__ == "__main__":
    unittest.main()
