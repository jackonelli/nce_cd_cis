import unittest
import torch

from src.ace.ace_cis_alt import AceCisAltCrit
from src.ace.ace_cis import AceCisCrit
from src.models.ace.ace_model import AceModel, ResidualBlock
from src.noise_distr.ace_proposal import AceProposal


class TestAceCisAlt(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceCisAltCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, 0)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

    def test_grad(self):
        # Check so that gradient is same as other ACE CIS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)

        crit = AceCisAltCrit(model, proposal, num_negative, energy_reg=0.1)
        crit_ref = AceCisCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        y_o, y_u, observed_mask = crit._mask_input(y)

        q, context = proposal.forward((y_o, observed_mask))
        y_samples = crit.inner_sample_noise(q, num_samples=num_negative)

        # Calculate grads.
        loss, _, _ = crit.inner_crit((y_u, observed_mask, context), (y_samples, q))
        loss.backward(retain_graph=True)
        grad_model = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal = [param.grad.detach().clone() for param in proposal.parameters()]

        # Calculate ref. grads.
        model.clear_gradients()
        proposal.clear_gradients()
        loss_ref, _, _ = crit_ref.inner_crit((y_u, observed_mask, context), (y_samples, q))
        loss_ref.backward()
        grad_model_ref = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal_ref = [param.grad.detach().clone() for param in proposal.parameters()]

        # Check parameters same
        for p, p_ref in zip(grad_model, grad_model_ref):
            assert torch.allclose(p, p_ref, rtol=1e-3)

        for p, p_ref in zip(grad_proposal, grad_proposal_ref):
            assert torch.allclose(p, p_ref, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
