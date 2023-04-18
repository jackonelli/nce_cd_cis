import unittest
import torch
from torch.distributions import Categorical

from src.aem.aem_pers_cis import AemCisJointPersCrit
from src.models.aem.made import ResidualMADE
from src.models.aem.energy_net import ResidualEnergyNet
from src.noise_distr.aem_proposal import AemProposal


class TestAemPersCis(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()
        num_mixture_components = 10
        num_samples = 100

        output_dim_multiplier = num_context_units + 3 * num_mixture_components

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))

        made = ResidualMADE(input_dim=num_features, n_residual_blocks=2, hidden_dim=10,
                            output_dim_multiplier=output_dim_multiplier)

        proposal = AemProposal(autoregressive_net=made, proposal_component_family='gaussian',
                               num_context_units=num_context_units, num_components=num_mixture_components)

        crit = AemCisJointPersCrit(model, proposal, num_negative, batch_size=num_samples)

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
        log_w_unnorm = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg + 1))
        sampled_idx = Categorical(logits=log_w_unnorm).sample()

        assert sampled_idx.shape == (num_samples,)

        y_p = torch.gather(ys, dim=1, index=sampled_idx[:, None, None].repeat(1, 1, num_features)).squeeze(dim=1)
        print(y_p.shape)
        assert y_p.shape == (num_samples, num_features)

        y_p_ref = torch.zeros((num_samples, num_features))
        for i in range(num_samples):
            for j in range(num_features):
                y_p_ref[i, j] = ys[i, sampled_idx[i], j]

        assert torch.allclose(y_p, y_p_ref)


if __name__ == "__main__":
    unittest.main()
