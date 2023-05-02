import unittest
import torch
from torch.distributions import Categorical

from src.aem.aem_smc import AemSmcCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemSmcCrit(unittest.TestCase):
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

        crit = AemSmcCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _, y_samples = crit.inner_crit(y)

        assert y_samples.shape == (num_samples, num_negative, num_features)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

    def test_sampling(self):
        # Check so that sampling step in SMC algorithm works as intended (or rather the gather function)

        num_samples = 100
        num_neg = 10
        num_features = 5

        y_s = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg, num_features))
        y_s_copy = y_s.clone()
        for i in range(1, num_features):
            log_w_tilde_y_s = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg))
            ancestor_inds = Categorical(logits=log_w_tilde_y_s).sample(sample_shape=torch.Size((num_neg,))).transpose(0,
                                                                                                                      1)
            assert ancestor_inds.shape == (num_samples, num_neg)
            y_s[:, :, :i] = torch.gather(y_s[:, :, :i], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i))

            y_s_ref = torch.zeros((num_samples, num_neg, i))
            for j in range(num_samples):
                for k in range(num_neg):
                    for l in range(i):
                        y_s_ref[j, k, l] = y_s_copy[j, ancestor_inds[j, k], l]

            assert torch.allclose(y_s[:, :, :i], y_s_ref)

            y_s_copy = y_s.clone()

    def test_proposal(self):
        """Check so that masking is correct in sampling"""

        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        y = torch.distributions.uniform.Uniform(low=1.0, high=10.0).sample((num_samples, num_features))

        class MockProposal:

            def __init__(self, num_features, num_context_units):
                self.num_features = num_features
                self.num_context_units = num_context_units

            def forward(self, y):
                res = torch.sum(y > 1e-6, dim=-1) / 2 + 1
                return res, res[:, None].repeat(1, self.num_context_units)

            def inner_sample(self, distr, num_samples):
                return distr

            def inner_log_prob(self, distr, y):
                return torch.ones(y.shape[0], )

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = MockProposal(num_features, num_context_units)

        crit = AemSmcCrit(model, proposal, num_negative)

        for i in range(num_features):
            _, _, y_samples = crit._proposal_log_probs(y, i, 0)

            assert y_samples.grad is None
            assert torch.allclose(y_samples[:, i], torch.tensor(i + 1).float().repeat(y_samples.shape[0]))

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

        crit = AemSmcCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        log_prob_p, log_prob_q, log_p_tilde_y = crit.log_prob(y, return_unnormalized=True)

        assert log_prob_p.shape == (num_samples,)
        assert log_prob_q.shape == (num_samples,)

        mean_log_prob_p, mean_log_prob_q = log_prob_p.mean(), log_prob_q.mean()
        assert not torch.isnan(mean_log_prob_p) or torch.isinf(mean_log_prob_p)
        assert not torch.isnan(mean_log_prob_q) or torch.isinf(mean_log_prob_q)
        assert log_prob_p.std() > 0.0
        assert log_prob_q.std() > 0.0

        log_p_tilde_y_ref = torch.zeros((num_samples, num_features))

        # Just a sanity check for reshape
        context = torch.zeros((y.shape[0], num_features, num_context_units))
        for i in range(num_features):
            _, context[:, i, :], _ = crit._proposal_log_probs(y, i, num_observed=y.shape[0])

        for i in range(num_features):
            log_p_tilde_y_ref[:, i] = crit._model_log_probs(y[:, i].reshape(-1, 1), context[:, i, :].reshape(-1, num_context_units))

        assert torch.allclose(log_p_tilde_y, log_p_tilde_y_ref)

if __name__ == "__main__":
    unittest.main()
