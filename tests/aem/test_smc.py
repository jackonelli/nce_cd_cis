import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
from src.aem.aem_smc import AemSmcCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.noise_distr.aem_proposal_joint_z_no_context import AemJointProposalWOContext


class TestSmc(unittest.TestCase):
    def test_smc(self):

        batch_size, num_samples, dim = 1, 10, 5

        num_res_blocks, num_hidden, num_components, num_context_units = 2, 5, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * dim, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemSmcCrit(model, proposal, num_samples)

        # SMC
        plot_dim = dim
        y_s = torch.zeros((batch_size, num_samples, dim))
        ess_all = torch.zeros((dim,))

        sampled_inds = torch.zeros((batch_size, num_samples, dim))
        sampled_inds[:, :, 0] = torch.arange(1, num_samples + 1, dtype=torch.float).reshape(1, -1).repeat(batch_size, 1)

        # First dim
        # Propagate
        log_q_y_s, context, y_s = crit._proposal_log_probs(y_s.reshape(-1, dim), 0, num_observed=0)
        context, y_s = context.reshape(-1, num_samples, crit.num_context_units), y_s.reshape(-1, num_samples, dim)

        # Reweight
        log_p_tilde_y_s = crit._model_log_probs(y_s[:, :, 0].reshape(-1, 1),
                                                context.reshape(-1, crit.num_context_units))
        del context

        log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_samples)
        del log_p_tilde_y_s, log_q_y_s

        # Dim 2 to D
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))

        for i in range(1, dim):
            # Resample
            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)

            ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach()
            ess_all[i - 1] = ess.mean(dim=0)

            resampling_inds = ess < (num_samples / 2)
            log_weight_factor = torch.zeros(log_w_tilde_y_s.shape)

            w = log_w_y_s.clone()
            if resampling_inds.sum() > 0:
                print("Resampling")
                with torch.no_grad():
                    ancestor_inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(
                        sample_shape=torch.Size((num_samples,))).transpose(0, 1)
                    print(torch.exp(log_w_y_s))

                    # Small test on sampling
                    inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(
                        sample_shape=torch.Size((10000,))).transpose(0, 1)
                    print(torch.bincount(inds.squeeze(), minlength=num_samples) / 10000)
                    assert ancestor_inds.shape == (batch_size, num_samples)

                    y_s_ref = y_s.clone()
                    y_s[resampling_inds, :, :i] = torch.gather(
                        y_s[resampling_inds, :, :i], dim=1,
                        index=ancestor_inds[:, :, None].repeat(1, 1, i))

                    w[resampling_inds, :] = torch.gather(log_w_y_s[resampling_inds, :], dim=1,
                                                         index=ancestor_inds[:, :])
                    sampled_inds[resampling_inds, :, i] = ancestor_inds.float() + 1.0
                    if i <= plot_dim:
                        print(i)
                        print("Test")
                        print(y_s[0, 1, :])
                        print(y_s_ref[0, ancestor_inds[0, 1], :])

                    y_s_old = y_s_ref.clone()
                    for j in range(batch_size):
                        for k in range(num_samples):
                            if resampling_inds[j]:
                                for l in range(i):
                                    y_s_ref[j, k, l] = y_s_old[j, ancestor_inds[j, k], l]

                    assert torch.allclose(y_s, y_s_ref)

            sampled_inds[~resampling_inds, :, i] = torch.arange(1, num_samples + 1, dtype=torch.float).reshape(1,
                                                                                                               -1).repeat(
                batch_size, 1)[~resampling_inds, :]

            if i <= plot_dim:
                for j in range(num_samples):  # There is probably an easier way to adapt the marker size
                    # print(torch.exp(log_w_y_s[:, j]))
                    plt.plot(i, j + 1, '.', markersize=100 * torch.exp(log_w_y_s[:, j]), c='blue', alpha=0.5)
                    plt.plot(i + 1, j + 1, '.', markersize=100 * torch.exp(w[:, j]), c='red', alpha=0.5)
                    plt.plot(np.array([i, i + 1]), np.array([sampled_inds[:, j, i].squeeze().cpu(), j + 1]), '-.',
                             c='orange')

            # if resampling_inds.sum() < batch_size:
            log_weight_factor[~resampling_inds, :] = log_w_y_s[~resampling_inds, :] + torch.log(
                torch.Tensor([num_samples]))
            del log_w_y_s, ess, resampling_inds

            # Propagate
            log_q_y_s, context, y_s = crit._proposal_log_probs(y_s.reshape(-1, dim), i, num_observed=0)
            context, y_s = context.reshape(-1, num_samples, crit.num_context_units), y_s.reshape(-1, num_samples,
                                                                                                 dim)

            # Reweight
            log_p_tilde_y_s = crit._model_log_probs(y_s[:, :, i].reshape(-1, 1),
                                                    context.reshape(-1, crit.num_context_units))

            log_q_y_s_ref = torch.zeros((batch_size, num_samples))
            log_p_tilde_y_s_ref = torch.zeros((batch_size, num_samples))

            for j in range(batch_size):
                for k in range(num_samples):
                    net_input = torch.cat(
                        (y_s[j, k, :].reshape(1, -1) * crit.mask[i, :], crit.mask[i, :].reshape(1, -1)),
                        dim=-1)
                    log_q_y_s_ref[j, k] = crit._noise_distr.log_prob(y_s[j, k, i].reshape(1, 1), net_input)
                    log_p_tilde_y_s_ref[j, k] = crit._model_log_probs(y_s[j, k, i].reshape(1, 1),
                                                                      context[j, k, :].reshape(1, crit.num_context_units))

            assert torch.abs(log_p_tilde_y_s - log_p_tilde_y_s_ref.reshape(batch_size, num_samples)).max() < 5e-5
            assert torch.allclose(log_p_tilde_y_s, log_p_tilde_y_s_ref.reshape(batch_size, num_samples), atol=5e-5)
            assert torch.abs(log_q_y_s - log_q_y_s_ref.reshape(batch_size, num_samples)).max() < 5e-5
            assert torch.allclose(log_q_y_s, log_q_y_s_ref.reshape(batch_size, num_samples), atol=5e-5)
            del context

            log_w_tilde_y_s = log_weight_factor + (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_samples)
            log_w_tilde_y_s_ref = log_weight_factor + log_p_tilde_y_s_ref - log_q_y_s_ref
            assert torch.abs(log_w_tilde_y_s - log_w_tilde_y_s_ref.reshape(batch_size, num_samples)).max() < 5e-5
            assert torch.allclose(log_w_tilde_y_s, log_w_tilde_y_s_ref.reshape(batch_size, num_samples), atol=5e-5)

            del log_p_tilde_y_s, log_q_y_s, log_weight_factor

            log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))

        log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
        ess_all[-1] = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach().mean(dim=0)

        if dim <= plot_dim:
            for j in range(num_samples):  # There is probably an easier way to adapt the marker size
                plt.plot(dim, j + 1, '.', markersize=100 * torch.exp(log_w_y_s[:, j]), c='blue', alpha=0.5)

        plt.show()

    def test_sampling(self):
        # test sampling on simple example
        dim, num_context_units = 10, 1
        num_samples = 1000


        mean = 1.0 #np.random.uniform(-5.0, 5.0)
        std = 1.8 #np.random.uniform(0.1, 2.0)

        class MockProposal:
            def __init__(self, num_features, num_context_units):
                self.num_features = num_features
                self.num_context_units = num_context_units
                self.counter = -1

            def forward(self, y):
                q = torch.distributions.normal.Normal(torch.zeros((y.shape[0])), 10 * torch.ones((y.shape[0])))

                if self.counter == -1:
                    context = y[:, 0].reshape(-1, 1)
                    self.counter += 1
                else:
                    context = y[:, self.counter].reshape(-1, 1)
                    self.counter += 1

                if self.counter == (self.num_features-1):
                    self.counter = -1

                return q, context

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
                return - 0.5 * (1 / self.std ** 2) * (y - context - self.mean) ** 2

        proposal = MockProposal(dim, num_context_units)
        model = MockModel(num_context_units, mean, std)
        crit = AemSmcCrit(model, proposal, num_samples)

        # Sample from model
        x = np.arange(-10, 10, 0.2)
        true_dens = torch.exp(torch.distributions.normal.Normal(mean, std).log_prob(torch.tensor(x))).numpy()

        reps = 100
        log_norm = torch.zeros((reps,))
        for r in range(reps):
            log_norm[r], y_s, log_w_tilde_y_s, _ = crit.inner_smc(1, num_samples, None)

        sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s.squeeze(dim=0)).sample((num_samples,))
        y = torch.gather(y_s.squeeze(dim=0), dim=0, index=sampled_idx[:, None].repeat(1, dim))

        print("Mean and std of log normalizer, {} reps: {}, {}".format(reps, log_norm.mean(), log_norm.std()))

        true_log_norm = (dim / 2) * torch.log(torch.tensor(2 * np.pi * std**2))
        print("True log normalizer: {}".format(true_log_norm))

        fig, ax = plt.subplots(1, dim)
        ax[0].hist(y[:, 0], density=True)
        ax[0].plot(x, true_dens)

        for i in range(dim-1):
            ax[i+1].hist(y[:, i+1] - y[:, i], density=True)
            ax[i+1].plot(x, true_dens)

        ax[0].set_title("Mean = {:.03}, std = {:.03}".format(mean, std))

        plt.show()

    def test_sampling_2(self):
        # Test sampling on simple example

        dim, num_context_units = 10, 1
        num_samples = 1000

        mean = 1.0  # np.random.uniform(-5.0, 5.0)
        std = 1.8  # np.random.uniform(0.1, 2.0)

        class MockProposal:
            def __init__(self, num_features, num_context_units):
                self.num_features = num_features
                self.num_context_units = num_context_units
                self.counter = -1

            def forward(self, y):
                q = torch.distributions.normal.Normal(torch.zeros((y.shape[0])), 10 * torch.ones((y.shape[0])))

                if self.counter == -1:
                    context = y[:, 0].reshape(-1, 1)
                    self.counter += 1
                else:
                    context = y[:, self.counter].reshape(-1, 1)
                    self.counter += 1

                if self.counter == (self.num_features - 1):
                    self.counter = -1

                return q, context

            def inner_sample(self, distr, size):
                return distr.sample(size).transpose(0, 1)

            def inner_log_prob(self, distr, samples):
                return distr.log_prob(samples.squeeze(dim=-1))

        class MockModel(torch.nn.Module):
            def __init__(self, num_context_units, mean, std, made):
                super().__init__()
                self.num_context_units = num_context_units
                self.mean = mean
                self.std = std
                self.made = made

            def forward(self, y):
                y_curr, context = y[:, :num_context_units], y[:, num_context_units:]
                mean = self.made.forward(y)

                return - 0.5 * (1 / self.std ** 2) * (y_curr - context - mean) ** 2

        proposal = MockProposal(dim, num_context_units)

        num_res_blocks, num_hidden = 2, 5
        model_made = ResidualMADEJoint(1 + num_context_units, num_res_blocks, num_hidden, 1)
        model = MockModel(num_context_units, mean, std, model_made)
        crit = AemSmcCrit(model, proposal, num_samples)

        # Sample from model
        x = np.arange(-10, 10, 0.2)
        true_dens = torch.exp(torch.distributions.normal.Normal(mean, std).log_prob(torch.tensor(x))).numpy()

        reps = 100
        log_norm = torch.zeros((reps,))
        for r in range(reps):
            log_norm[r], y_s, log_w_tilde_y_s, _ = crit.inner_smc(1, num_samples, None)

        sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s.squeeze(dim=0)).sample((num_samples,))
        y = torch.gather(y_s.squeeze(dim=0), dim=0, index=sampled_idx[:, None].repeat(1, dim))

        print("Mean and std of log normalizer, {} reps: {}, {}".format(reps, log_norm.mean(), log_norm.std()))

        true_log_norm = (dim / 2) * torch.log(torch.tensor(2 * np.pi * std ** 2))
        print("True log normalizer: {}".format(true_log_norm))

        fig, ax = plt.subplots(1, dim)
        ax[0].hist(y[:, 0], density=True)
        ax[0].plot(x, true_dens)

        for i in range(dim - 1):
            # with torch.no_grad():
            #    mean_ind = model_made.forward(torch.cat((y[:, i + 1].reshape(-1, 1), y[:, i].reshape(-1, 1)), dim=1))
            mean_ind = 0

            ax[i + 1].hist(y[:, i + 1] - y[:, i] - mean_ind, density=True)
            ax[i + 1].plot(x, true_dens)

        ax[0].set_title("Mean = {:.03}, std = {:.03}".format(mean, std))

        plt.show()

    def test_sampling_3(self):
        # Test sampling on simple example where we use context model

        dim = 10
        num_samples = 100000

        mean = 1.0  # np.random.uniform(-5.0, 5.0)
        std = 1.8  # np.random.uniform(0.1, 2.0)

        num_context_units = 1
        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = 3 * num_components

        class MockProposal:
            def __init__(self, num_features, num_context_units, proposal_model):
                self.num_features = num_features
                self.num_context_units = num_context_units
                self.counter = -1
                self.proposal_model = proposal_model

            def forward(self, y):
                q, context_full = self.proposal_model.forward(y)

                if self.counter == -1:
                    context = context_full[:, 0].reshape(-1, 1)
                    assert torch.allclose(context, y[:, 0].reshape(-1, 1))
                    self.counter += 1
                else:
                    context = context_full[:, self.counter].reshape(-1, 1)
                    assert torch.allclose(context, y[:, self.counter].reshape(-1, 1))
                    self.counter += 1

                if self.counter == (self.num_features - 1):
                    self.counter = -1

                return q, context

            def inner_sample(self, distr, size):
                return distr.sample(size).transpose(0, 1)

            def inner_log_prob(self, distr, samples):
                return distr.log_prob(samples)#.squeeze(dim=-1))

        class MockModel(torch.nn.Module):
            def __init__(self, num_context_units, mean, std, made):
                super().__init__()
                self.num_context_units = num_context_units
                self.mean = mean
                self.std = std
                self.made = made

            def forward(self, y):
                y_curr, context = y[:, :num_context_units], y[:, num_context_units:]
                mean = self.made.forward(y)
                return - 0.5 * (1 / self.std ** 2) * (y_curr - context - mean) ** 2

        made = ResidualMADEJoint(2 * dim, num_res_blocks, num_hidden, output_dim_mult)
        proposal_model = AemJointProposalWOContext(made, num_components)
        proposal = MockProposal(dim, num_context_units, proposal_model)

        model_made = ResidualMADEJoint(1 + num_context_units, num_res_blocks, num_hidden, 1)
        model = MockModel(num_context_units, mean, std, model_made)
        crit = AemSmcCrit(model, proposal, num_samples)

        # Sample from model
        x = np.arange(-10, 10, 0.2)
        true_dens = torch.exp(torch.distributions.normal.Normal(mean, std).log_prob(torch.tensor(x))).numpy()

        reps = 100
        log_norm = torch.zeros((reps,))
        for r in range(reps):
            log_norm[r], y_s, log_w_tilde_y_s, _ = crit.inner_smc(1, num_samples, None)

        sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s.squeeze(dim=0)).sample((num_samples,))
        y = torch.gather(y_s.squeeze(dim=0), dim=0, index=sampled_idx[:, None].repeat(1, dim))

        print("Mean and std of log normalizer, {} reps: {}, {}".format(reps, log_norm.mean(), log_norm.std()))

        true_log_norm = (dim / 2) * torch.log(torch.tensor(2 * np.pi * std**2))
        print("True log normalizer: {}".format(true_log_norm))

        fig, ax = plt.subplots(1, dim)
        ax[0].hist(y[:, 0], density=True)
        ax[0].plot(x, true_dens)

        for i in range(dim-1):
            ax[i+1].hist(y[:, i+1] - y[:, i], density=True)
            ax[i+1].plot(x, true_dens)

        ax[0].set_title("Mean = {:.03}, std = {:.03}".format(mean, std))

        plt.show()

if __name__ == "__main__":
    unittest.main()
    #TestSmc().test_sampling_3()
