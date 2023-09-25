import unittest
import torch

from src.ace.ace_pers_cis import AceCisPers
from src.models.ace.ace_model import AceModel
from src.noise_distr.ace_proposal import AceProposal


class TestAcePersCis(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceCisPers(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, torch.arange(0, num_samples))
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)


if __name__ == "__main__":
    unittest.main()
