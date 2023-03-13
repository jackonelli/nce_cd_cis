import unittest
import torch
from torch.distributions import Categorical

from src.ace.ace_pers_cis import AceCisPers
from src.models.ace.ace_model import AceModel
from src.noise_distr.ace_proposal import AceProposal


class TestAcePersCis(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceCisPers(model, proposal, num_negative, energy_reg=0.1, batch_size=num_samples)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, num_samples)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

    def test_sampling(self):

        num_samples = 100
        num_neg = 10
        num_features = 5

        ys = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg + 1, num_features))
        log_w_unnorm = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg + 1, num_features))
        sampled_idx = Categorical(logits=log_w_unnorm.transpose(1, 2)).sample()

        assert sampled_idx.shape == (num_samples, num_features)

        y_p = torch.gather(ys, dim=1, index=sampled_idx.unsqueeze(dim=1)).squeeze(dim=1)
        assert y_p.shape == (num_samples, num_features)

        y_p_ref = torch.zeros((num_samples, num_features))
        for i in range(num_samples):
            for j in range(num_features):
                y_p_ref[i, j] = ys[i, sampled_idx[i, j], j]

        assert torch.allclose(y_p, y_p_ref)

        y_p_ref_2 = torch.gather(ys.transpose(1, 2), dim=-1, index=sampled_idx.unsqueeze(dim=-1)).squeeze(dim=-1)
        assert torch.allclose(y_p, y_p_ref_2)


if __name__ == "__main__":
    unittest.main()
