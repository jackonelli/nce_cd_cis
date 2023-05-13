import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemCisCrit(unittest.TestCase):
    def test_crit(self):
        # Sanity check comparing with IS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((1,)).repeat(num_samples, 1)

        class MockProposal(AemJointProposal):
            """Mock proposal that always samples y"""

            def __init__(self, autoregressive_net, num_context_units, num_components, y, num_negative):
                super().__init__(autoregressive_net, num_context_units, num_components)
                self._y = y
                self._num_neg = num_negative
                self.counter = 0

            def inner_sample(self, distr, num_samples):
                # Always return y (weights will be the same for all obs.)

                if self.counter >= y.shape[-1]:
                    self.counter = 0

                self.counter += 1
                return self._y[:, self.counter - 1].reshape(-1, 1, 1).repeat(1, self._num_neg + 1, 1).reshape(-1, 1)

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = MockProposal(made, num_context_units, num_components, y, num_negative)

        crit = AemCisJointCrit(model, proposal, num_negative)
        crit_ref = AemIsJointCrit(model, proposal, num_negative)

        _, loss_p, loss_q, y_samples = crit.inner_crit(y)

        assert y_samples.grad is None

        _, loss_p_ref, loss_q_ref, y_samples_ref = crit_ref.inner_crit(y)

        y_s = y_samples.reshape(-1, num_negative, num_features)
        assert torch.allclose(y_s[:, 0, :], y)
        assert torch.allclose(y_samples, y_samples_ref)
        assert torch.allclose(loss_q, loss_q_ref)
        assert torch.allclose(loss_p, loss_p_ref)

        assert y_samples.shape == (num_samples * num_negative, num_features)

    def test_sampling(self):
        num_features, num_context_units = 2, 1
        num_negative = torch.randint(low=2, high=5, size=torch.Size((1,))).item()
        num_samples = 100

        mean = np.random.uniform(-5.0, 5.0)
        std = np.random.uniform(0.1, 2.0)

        class MockProposal:
            def __init__(self, num_features, num_context_units):
                self.num_features = num_features
                self.num_context_units = num_context_units

            def forward(self, y):
                q = torch.distributions.normal.Normal(y[:, 0], 10 * torch.ones((y.shape[0])))
                return q, y[:, 0].reshape(-1, 1)

            def inner_sample(self, distr, size):
                return distr.sample(size).transpose(0, 1)

            def inner_log_prob(self, distr, samples):
                return distr.log_prob(samples.squeeze(dim=-1))

        class MockModel(torch.nn.Module):
            def __init__(self, num_context_units, mean, std):
                super().__init__()
                self.num_context_units = num_context_units
                self.mean = mean
                self.std = std

            def forward(self, y):
                y, context = y[:, :num_context_units], y[:, num_context_units:]
                return - 0.5 * (1 / self.std**2) * (y-context-self.mean)**2

        proposal = MockProposal(num_features, num_context_units)
        model = MockModel(num_context_units, mean, std)
        crit = AemCisJointCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        # Sample from model
        num_steps = 1000
        check_points = [1, 10, 100, 1000]
        fig, ax = plt.subplots(2, len(check_points))
        count = 0

        x = np.arange(-10, 10, 0.2)
        true_dens = torch.exp(torch.distributions.normal.Normal(mean, std).log_prob(torch.tensor(x))).numpy()

        with torch.no_grad():
            for t in range(num_steps):
                _, _, _, log_w_tilde_y_s, _, y_samples = crit._log_probs(y, num_negative)
                y_s = torch.cat((y.reshape(-1, 1, num_features), y_samples.reshape(-1, num_negative, num_features)), dim=1)
                sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s).sample()
                y = torch.gather(y_s, dim=1, index=sampled_idx[:, None, None].repeat(1, 1, y.shape[-1])).squeeze(dim=1)

                if (t+1) in check_points:
                    ax[0, count].hist(y[:, 0], density=True)
                    ax[0, count].plot(x, true_dens)
                    ax[1, count].hist(y[:, 1] - y[:, 0], density=True)
                    ax[1, count].plot(x, true_dens)
                    ax[0, count].set_title("It. {}. Mean = {:.03}, std = {:.03}".format(t + 1, mean, std))
                    count += 1

        plt.show()

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

        crit = AemCisJointCrit(model, proposal, num_negative)

        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        log_prob_p, log_prob_q = crit.log_prob(y)

        assert log_prob_p.shape == (num_samples,)
        assert log_prob_q.shape == (num_samples,)

        mean_log_prob_p, mean_log_prob_q = log_prob_p.mean(), log_prob_q.mean()
        assert not torch.isnan(mean_log_prob_p) or torch.isinf(mean_log_prob_p)
        assert not torch.isnan(mean_log_prob_q) or torch.isinf(mean_log_prob_q)
        assert log_prob_p.std() > 0.0
        assert log_prob_q.std() > 0.0


if __name__ == "__main__":
    unittest.main()
