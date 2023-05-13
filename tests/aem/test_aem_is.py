import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal

from src.models.aem.energy_net import ResidualBlock


class TestAemIsCrit(unittest.TestCase):
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

        crit = AemIsJointCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _, y_samples = crit.inner_crit(y)

        assert y_samples.shape == (num_samples * num_negative, num_features)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

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

        crit = AemIsJointCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        log_prob_p, log_prob_q, log_prob_p_tilde = crit.log_prob(y, return_unnormalized=True)

        assert log_prob_p.shape == (num_samples,)
        assert log_prob_q.shape == (num_samples,)

        mean_log_prob_p, mean_log_prob_q = log_prob_p.mean(), log_prob_q.mean()
        assert not torch.isnan(mean_log_prob_p) or torch.isinf(mean_log_prob_p)
        assert not torch.isnan(mean_log_prob_q) or torch.isinf(mean_log_prob_q)
        assert log_prob_p.std() > 0.0
        assert log_prob_q.std() > 0.0

        # Just check so that this really is log prob of y
        shuffled_inds = torch.randperm(y.shape[0])
        y_1 = y[shuffled_inds, :]

        _, _, log_prob_p_tilde_ref = crit.log_prob(y_1, return_unnormalized=True)

        log_prob_p_tilde, _ = torch.sort(log_prob_p_tilde, dim=0)
        log_prob_p_tilde_ref, _ = torch.sort(log_prob_p_tilde_ref, dim=0)
        assert torch.allclose(log_prob_p_tilde, log_prob_p_tilde_ref)

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

        crit = AemIsJointCrit(model, proposal, num_negative)

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
        plt.show()

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
                return torch.ones(y.shape[0],)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = MockProposal(num_features, num_context_units)

        crit = AemIsJointCrit(model, proposal, num_negative)

        _, _, _, y_samples = crit.inner_crit(y)

        assert y_samples.grad is None

        for i in range(num_features):
            assert torch.allclose(y_samples[:, i], torch.tensor(i+1).float().repeat(y_samples.shape[0]))

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

        crit = AemIsJointCrit(model, proposal, num_negative)

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

        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                pass
                #assert torch.allclose(param.grad, torch.tensor(0.0))
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

    def test_eval(self):
        # Check so that model.eval() turns of dropout+batchnorm correctly
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        dropout_rate = 0.8
        use_bn = True
        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult, use_batch_norm=use_bn,
                                 dropout_probability=dropout_rate, zero_initialization=False)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1), use_batch_norm=use_bn,
                                  dropout_probability=dropout_rate, zero_initialization=False)
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemIsJointCrit(model, proposal, num_negative, num_neg_samples_validation=2)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        # Run once without eval
        _, log_prob_q_1, log_prob_p_1 = crit.log_prob(y, return_unnormalized=True)
        # Use eval
        model.eval()
        made.eval()

        _, log_prob_q_2, log_prob_p_2 = crit.log_prob(y, return_unnormalized=True)

        # These should not be the same if dropout_rate > 0 (although it might not be impossible)
        if dropout_rate > 0.0 or use_bn:
            assert not torch.allclose(log_prob_p_1, log_prob_p_2)
            assert not torch.allclose(log_prob_q_1, log_prob_q_2)

        # Run again to see that we get the same res.
        _, log_prob_q_3, log_prob_p_3 = crit.log_prob(y, return_unnormalized=True)

        # This should give the same output
        assert torch.allclose(log_prob_p_2, log_prob_p_3)
        assert torch.allclose(log_prob_q_2, log_prob_q_3)

        # If we go back to train mode
        model.train()
        made.train()

        _, log_prob_q_4, log_prob_p_4 = crit.log_prob(y, return_unnormalized=True)
        if dropout_rate > 0.0 or use_bn:
            assert not torch.allclose(log_prob_q_3, log_prob_p_4)
            assert not torch.allclose(log_prob_q_3, log_prob_q_4)



if __name__ == "__main__":
    unittest.main()
