import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from src.aem.aem_smc import AemSmcCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal

from src.models.aem.energy_net import ResidualBlock


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

        # Resampling always
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

        # Resampling only when ess falls below num_chains/2
        y_s_copy = y_s.clone()

        for i in range(1, num_features):
            log_w_tilde_y_s = torch.distributions.normal.Normal(loc=0.0, scale=1.0).sample((num_samples, num_neg))

            # Just a sanity check for calculating the ESS
            ess = torch.exp(- torch.logsumexp(2 * (log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)), dim=1))

            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
            ess_ref = 1 / torch.sum(torch.exp(log_w_y_s)**2, dim=1)

            assert torch.allclose(ess, ess_ref)

            resampling_inds = ess < (num_neg / 2)

            ancestor_inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(
                sample_shape=torch.Size((num_neg,))).transpose(0, 1)
            assert ancestor_inds.shape == (int(resampling_inds.sum()), num_neg)

            y_s[resampling_inds, :, :i] = torch.gather(y_s[resampling_inds, :, :i], dim=1,
                                                       index=ancestor_inds[:, :, None].repeat(1, 1, i))

            y_s_ref = y_s_copy[:, :, :i].clone()
            counter = 0
            for j in range(num_samples):
                if ess[j] < (num_neg / 2):
                    assert resampling_inds[j]
                    for k in range(num_neg):
                        for l in range(i):
                            y_s_ref[j, k, l] = y_s_copy[j, ancestor_inds[counter, k], l]
                    counter += 1
                else:
                    assert not resampling_inds[j]

            assert counter == resampling_inds.sum()
            assert torch.allclose(y_s[:, :, :i], y_s_ref)

            y_s_copy = y_s.clone()

            # Additional sanity check
            log_w_tilde_y_s = torch.Tensor([[-100000, 100000], [1000000, -1000000], [100, 100]])
            ancestor_inds = Categorical(logits=log_w_tilde_y_s).sample(sample_shape=torch.Size((num_neg,))).transpose(0, 1)
            assert torch.allclose(ancestor_inds[0, :], torch.ones((num_neg,), dtype=torch.long))
            assert torch.allclose(ancestor_inds[1, :], torch.zeros((num_neg,), dtype=torch.long))

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

    def test_part_fn(self):
        # Check so that part. fun. is calculated as expected

        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemSmcCrit(model, proposal, num_negative)

        model.eval()
        made.eval()
        crit.set_training(False)

        rep = 10
        num_neg = [5, 10, 100, 500, 1000, 5000, 10000, 50000, 100000]
        log_norm = torch.zeros((len(num_neg), rep))
        for i, j in enumerate(num_neg):
            crit.set_num_proposal_samples_validation(j)
            for k in range(rep):
                with torch.no_grad():
                    log_norm[i, k] = crit.log_part_fn()

        plt.errorbar(np.array(num_neg), log_norm.mean(dim=-1), yerr=log_norm.std(dim=-1))
        print(log_norm.std(dim=-1))
        plt.title("SMC log-normalizer estimate")
        plt.show()

    def test_grad(self):
        # Check so that gradients of both proposal and model are updated in gradient update
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

        # Update only for model
        crit.calculate_crit_grad_p(y, 0)

        # Check that parameters of model have been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            else:
                print("Note: param is not torch.nn.Parameter")

        for param in made.parameters():
            if isinstance(param, torch.nn.Parameter):
                #assert param.grad is None
                pass

            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

        # Update only for proposal
        model.clear_gradients()
        made.clear_gradients()
        crit.calculate_crit_grad_q(y, 0)

        # Check that parameters of model have NOT been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert torch.allclose(param.grad, torch.tensor(0.0))
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

    def test_reshape(self):
        # Sanity check on use of reshape

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

        # First check
        y_samples = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples, num_negative))

        y_s = torch.cat((y.unsqueeze(dim=1), y_samples), dim=1)
        y_ref = y_s.clone().reshape(-1, num_features)

        for i in range(num_samples):
            assert torch.abs(y_ref[i * (num_negative + 1), :]-y_s[i, 0, :]).max() < 1e-4
            assert torch.allclose(y_s[i, 0, :], y_ref[i * (num_negative + 1), :], atol=1e-5)

        # Second check
        selected_dim = torch.randint(0, num_features, (1,))[0]
        log_q_y_s, context, y_s_ref = crit._proposal_log_probs(y_s.reshape(-1, num_features), selected_dim, num_observed=y.shape[0])
        context, y_s_ref = context.reshape(-1, num_negative + 1, num_context_units), y_s_ref.reshape(-1, num_negative + 1, num_features)

        assert torch.allclose(y_s[:, :, :selected_dim], y_s_ref[:, :, :selected_dim])

        context_ref = torch.zeros((num_samples, num_negative + 1, num_context_units))
        for i in range(num_samples):
            for j in range(num_negative + 1):
                _, context_ref[i, j, :], _ = crit._proposal_log_probs(y_s[i, j, :].reshape(-1, num_features), selected_dim, num_observed=1)

        assert torch.abs(context_ref-context).max() < 1e-5
        assert torch.allclose(context, context_ref, atol=1e-5)

        # Third check
        log_q_y_s, context, y_s_ref = crit._proposal_log_probs(y_s.reshape(-1, num_features), selected_dim, num_observed=y_s.shape[0])
        log_p_tilde_y_s = crit._model_log_probs(y_s[:, :, selected_dim].reshape(-1, 1), context.reshape(-1, num_context_units))
        log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_negative + 1)

        log_q_y_s_ref, log_p_tilde_y_s_ref = torch.zeros((num_samples, num_negative + 1)), \
                                             torch.zeros((num_samples, num_negative + 1))
        log_w_tilde_y_s_ref = torch.zeros((num_samples, num_negative + 1))
        for i in range(num_samples):
            for j in range(num_negative + 1):
                log_q_y_s_ref[i, j], context, y_s_ref = crit._proposal_log_probs(y_s[i, j, :].reshape(-1, num_features), selected_dim, num_observed=y_s.shape[0])
                log_p_tilde_y_s_ref[i, j] = crit._model_log_probs(y_s_ref[:, selected_dim].reshape(-1, 1), context.reshape(-1, num_context_units))
                log_w_tilde_y_s_ref[i, j] = log_p_tilde_y_s_ref[i, j] - log_q_y_s_ref[i, j]

        log_q_y_s, log_p_tilde_y_s = log_q_y_s.reshape(-1, num_negative + 1), log_p_tilde_y_s.reshape(-1, num_negative + 1)
        assert torch.abs(log_q_y_s_ref-log_q_y_s).max() < 1e-5
        assert torch.allclose(log_q_y_s, log_q_y_s_ref, atol=1e-5)
        assert torch.abs(log_p_tilde_y_s_ref-log_p_tilde_y_s).max() < 1e-5
        assert torch.allclose(log_p_tilde_y_s, log_p_tilde_y_s_ref, atol=1e-5)
        assert torch.abs(log_w_tilde_y_s_ref-log_w_tilde_y_s).max() < 1e-5
        assert torch.allclose(log_w_tilde_y_s, log_w_tilde_y_s_ref, atol=1e-5)

        y_samples = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples, num_negative))

        y_samples_ref = y_samples[:, 0, :].clone()
        log_p_tilde_y_samples = crit._model_log_probs(y_samples.reshape(-1, 1), torch.ones(num_samples * num_negative
                                                                                           * num_features, num_context_units)).reshape(-1, num_negative, num_features)

        log_p_tilde_y_samples_ref = crit._model_log_probs(y_samples_ref.reshape(-1, 1), torch.ones(num_samples * num_features,
                                                                                           num_context_units)).reshape(-1, num_features)

        assert torch.allclose(log_p_tilde_y_samples[:, 0, :], log_p_tilde_y_samples_ref)

if __name__ == "__main__":
    unittest.main()
