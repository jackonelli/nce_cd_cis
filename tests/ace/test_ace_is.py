import unittest
import torch

from src.nce.ace_is import AceIsCrit
from src.models.ace.ace_model import AceModel, ResidualBlock
from src.noise_distr.ace_proposal import AceProposal


class TestAceIs(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceIsCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, 0)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)


    def test_grad(self):
        # Check so that gradients of both proposal and model are updated in gradient update

        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceIsCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        # Update only for model
        crit.calculate_crit_grad_p(y, 0)

        # Check that parameters of model have been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            else:
                print("Note: param is not torch.nn.Parameter")

        # Check that parameters of proposal have NOT been updated
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is None
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

        # Update only for proposal
        model.clear_gradients()
        crit.calculate_crit_grad_q(y, 0)

        # Check that parameters of model have NOT been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is None
            else:
                print("Note: param is not torch.nn.Parameter")

        # Check that parameters of proposal have been updated
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

        # Update for both model and proposal
        model.clear_gradients()
        crit.calculate_crit_grad(y, 0)

        # Check that parameters of model have been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            else:
                print("Note: param is not torch.nn.Parameter")

            # Check that parameters of proposal have been updated
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")


if __name__ == "__main__":
    unittest.main()