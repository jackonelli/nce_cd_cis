# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.models.aem.made_joint_z import get_autoregressive_mask


class AemIsJointCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int = 1e2,
                 alpha: float = 1.0):

        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.alpha = alpha
        self.dim = self._noise_distr.num_features
        self.num_context_units = self._noise_distr.num_context_units
        self.mask = get_autoregressive_mask(self.dim)
        self.num_neg_samples_validation = int(num_neg_samples_validation)
        self.training = True
        self.counter = 0

    def crit(self, y, _idx):
        loss, p_loss, q_loss, _ = self.inner_crit(y)

        return loss, p_loss, q_loss

    def inner_crit(self, y: Tensor, y_samples: Tensor = None):

        if self.training:
            self.counter += 1

        # Calculate (unnormalized) densities
        log_p_tilde_y, log_q_y, log_normalizer, y_samples = self._log_probs(y, self._num_neg)

        # Calculate normalized density
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_samples

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()
        self._noise_distr.autoregressive_net.clear_gradients()

        # This should automatically assign gradients to model parameters
        loss, lp, lq = self.crit(y, _idx)
        loss.backward()

    def calculate_crit_grad_p(self, y: Tensor, _idx: Optional[Tensor]):
        # Entry for testing

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        _, p_loss, _ = self.crit(y, _idx)
        p_loss.backward()

    def calculate_crit_grad_q(self, y: Tensor, _idx: Optional[Tensor]):
        # Entry for testing

        # Clear gradients to avoid any issues
        self._noise_distr.autoregressive_net.clear_gradients()

        # This should automatically assign gradients to model parameters
        _, _, q_loss = self.crit(y, _idx)
        q_loss.backward()

    def _log_probs(self, y, num_samples):

        # Calculate (unnormalized) densities
        log_q_y, log_q_y_samples, context, y_samples = self._proposal_log_probs(y, num_samples=num_samples)
        # context = context.detach()
        log_p_tilde_y_s = torch.sum(self._model_log_probs(torch.cat((y, y_samples.detach()), dim=0).reshape(-1, 1),
                                                          context.reshape(-1, self.num_context_units)).reshape(-1,
                                                                                                               self.dim),
                                    dim=-1)

        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_y_s[:y.shape[0]], log_p_tilde_y_s[y.shape[0]:]

        # Calculate log normalizer
        log_normalizer = torch.logsumexp((log_p_tilde_y_samples - log_q_y_samples.detach()).reshape(-1, num_samples),
                                         # stop gradient
                                         dim=1) - torch.log(torch.Tensor([num_samples]))

        return log_p_tilde_y, log_q_y, log_normalizer, y_samples

    def _proposal_log_probs_dim(self, y, dim: int, num_observed: int = 0):
        net_input = torch.cat(
            (y * self.mask[dim, :], self.mask[dim, :].reshape(1, -1).repeat(y.shape[0], 1)),
            dim=-1)

        q_i, context = self._noise_distr.forward(net_input)

        if num_observed < y.shape[0]:
            with torch.no_grad():
                # OBS: VARFÖR HADE JAG SQUEEZE = 0 INNAN???
                samp = self._noise_distr.inner_sample(q_i, torch.Size((1,)))
                y[num_observed:, dim] = samp.squeeze(dim=0)[
                                        num_observed:]
                print(samp.max())

        log_q_y_s = self._noise_distr.inner_log_prob(q_i, y[:, dim].unsqueeze(dim=-1)).squeeze()

        return log_q_y_s, context, y

    def _proposal_log_probs(self, y, num_samples: int):
        y_samples = torch.zeros((y.shape[0] * num_samples, self.dim))
        context = torch.zeros((y.shape[0] * (num_samples + 1), self.dim, self.num_context_units))
        log_q_y_s = torch.zeros((y.shape[0] * (num_samples + 1), self.dim))

        # y_s = torch.cat((y * self.mask[0, :], y_samples))
        y_s = torch.cat((y, y_samples))
        for i in range(self.dim):
            log_q_y_s[:, i], context[:, i, :], y_s = self._proposal_log_probs_dim(y_s, i, y.shape[0])
            print(y_s.max())
            # y_s = torch.cat((y * self.mask[i + 1, :], y_samples))

        log_q_y_s = log_q_y_s.sum(dim=-1)
        log_q_y, log_q_y_samples = log_q_y_s[:y.shape[0]], log_q_y_s[y.shape[0]:]
        y_samples = y_s[y.shape[0]:, :]

        return log_q_y, log_q_y_samples, context, y_samples

    # def _proposal_log_probs(self, y, num_samples: int):
    # y_samples = torch.zeros((y.shape[0] * num_samples, self.dim))
    # context = torch.zeros((y.shape[0] * (num_samples + 1), self.dim, self.num_context_units))
    # log_q_y_s = torch.zeros((y.shape[0] * (num_samples + 1), self.dim))

    # y_s = torch.cat((y * self.mask[0, :], y_samples))
    # for i in range(self.dim):
    # net_input = torch.cat((y_s, self.mask[i, :].reshape(1, -1).repeat(y_s.shape[0], 1)), dim=-1)
    # q_i, context[:, i, :] = self._noise_distr.forward(net_input)

    # # if np.mod(self.counter, 100) == 0:
    # # print(i)
    # # print(q_i.components_distribution.scale.min())
    # # print(q_i.components_distribution.scale.median())
    # # print(q_i.components_distribution.scale.max())
    # # print(q_i.components_distribution.scale[0])
    # # print(q_i.components_distribution.scale[-1])

    # with torch.no_grad():
    # y_samples[:, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()[y.shape[0]:]

    # y_s = torch.cat((y * self.mask[i + 1, :], y_samples))
    # log_q_y_s[:, i] = self._noise_distr.inner_log_prob(q_i, y_s[:, i].unsqueeze(dim=-1)).squeeze()

    # log_q_y_s = log_q_y_s.sum(dim=-1)
    # log_q_y, log_q_y_samples = log_q_y_s[:y.shape[0]], log_q_y_s[y.shape[0]:]

    # return log_q_y, log_q_y_samples, context, y_samples

    # def _proposal_log_probs(self, y, num_samples: int):
    #
    #     y_samples = torch.zeros((y.shape[0], num_samples, self.dim))
    #     context = torch.zeros((y.shape[0], (num_samples + 1), self.dim, self.num_context_units))
    #     log_q_y_s = torch.zeros((y.shape[0], (num_samples + 1), self.dim))
    #
    #     y_s = torch.cat((y.unsqueeze(dim=1), y_samples), dim=1)
    #     for i in range(self.dim):
    #         for b in range(num_samples + 1):
    #             net_input = torch.cat((y_s[:, b, :] * self.mask[i, :], self.mask[i, :].reshape(1, -1).repeat(y.shape[0], 1)), dim=-1)
    #             q_i, context[:, b, i, :] = self._noise_distr.forward(net_input)
    #
    #             if b > 0:
    #                 with torch.no_grad():
    #                     y_samples[:, b-1, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()
    #
    #             log_q_y_s[:, b, i] = self._noise_distr.inner_log_prob(q_i, y_s[:, b, i].unsqueeze(dim=-1)).squeeze()
    #
    #     context_y, context_y_samples = context[:, 0, ::].reshape(-1, y.shape[-1]), context[:, 1:, ::].reshape(-1, y.shape[-1])
    #     log_q_y, log_q_y_samples =log_q_y_s[:, 0, :].sum(dim=-1).reshape(-1), log_q_y_s[:, 1:, :].sum(dim=-1).reshape(-1)
    #
    #     return log_q_y, log_q_y_samples, context_y, context_y_samples, y_samples.reshape(-1, y.shape[-1])

    def _model_log_probs(self, y, context):

        energy_net_inputs = torch.cat(
            (y, context),
            dim=-1
        )

        del y, context  # free GPU memory

        # Inputs to energy net can have very large batch size since we evaluate all importance samples at once.
        # We must split a very large batch to avoid OOM errors
        if energy_net_inputs.shape[0] > 300000 and not self.training:
            batch_size = 300000
            n_batches, leftover = (
                energy_net_inputs.shape[0] // batch_size,
                energy_net_inputs.shape[0] % batch_size
            )
            slices = [slice(batch_size * i, batch_size * (i + 1)) for i in
                      range(n_batches)]
            slices.append(
                slice(batch_size * n_batches, batch_size * n_batches + leftover))
            energy_net_outputs = torch.cat(
                [self._unnorm_distr(energy_net_inputs[slice_]).detach()  # stop gradient
                 for slice_ in slices],
                dim=0
            )
        else:
            energy_net_outputs = self._unnorm_distr(energy_net_inputs)

        del energy_net_inputs  # free GPU memory

        energy_net_outputs = energy_net_outputs.sum(dim=-1)

        # unnormalized log densities given by energy net
        return energy_net_outputs

    def log_prob(self, y, return_unnormalized=False):

        # Calculate (unnormalized) densities
        log_p_tilde_y, log_q_y, log_normalizer, _ = self._log_probs(y, self.num_neg_samples_validation)

        # Calculate/estimate normalized densities
        log_prob_p = log_p_tilde_y - log_normalizer
        log_prob_q = log_q_y

        if return_unnormalized:
            return log_prob_p, log_prob_q, log_p_tilde_y
        else:
            return log_prob_p, log_prob_q

    def sample(self, batch_size=1000):

        num_proposal_samples = 100
        y_samples = torch.zeros(batch_size * num_proposal_samples, self.dim)

        for i in range(self.dim):
            assert torch.allclose(y_samples, y_samples * (1 - self.mask[i, :]))
            net_input = torch.cat((y_samples, self.mask[i, :].reshape(1, -1).repeat(y_samples.shape[0], 1)), dim=-1)
            q_i, _ = self._noise_distr.forward(net_input)

            with torch.no_grad():
                y_samples[:, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()

        return y_samples.reshape(batch_size, num_proposal_samples, -1)

    def unnorm_log_prob(self, y):

        log_p_tilde_y, log_q_y = self._unnorm_log_prob(y)

        assert log_q_y.shape == (y.shape[0],)
        assert log_p_tilde_y.shape == (y.shape[0],)

        return log_p_tilde_y, log_q_y

    def part_fn(self, y, y_samples) -> Tensor:
        return 0

    def log_part_fn(self, return_ess=False):

        y_samples = torch.zeros((self.num_neg_samples_validation, self.dim))
        log_p_tilde_y_samples, log_q_y_samples = self._unnorm_log_prob(y_samples, sample=True)

        assert log_q_y_samples.shape == (self.num_neg_samples_validation,)
        assert log_p_tilde_y_samples.shape == (self.num_neg_samples_validation,)

        log_normalizer = torch.logsumexp((log_p_tilde_y_samples - log_q_y_samples.detach()),
                                         # stop gradient
                                         dim=0) - torch.log(torch.Tensor([self.num_neg_samples_validation]))
        if return_ess:
            log_w_tilde_y_s = log_p_tilde_y_samples - log_q_y_samples.detach()
            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=0, keepdim=True)
            ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=0)).detach()

            return log_normalizer, ess

        else:
            return log_normalizer

    def _unnorm_log_prob(self, y, sample=False, return_samples=False):
        # TODO: this is a bit unnecessary, but here we do not need to draw samples
        # Calculate (unnormalized) densities for y

        log_q_y, log_p_tilde_y = torch.zeros((y.shape[0],)), torch.zeros((y.shape[0],))
        for i in range(self.dim):
            net_input = torch.cat(
                (y * self.mask[i, :], self.mask[i, :].reshape(1, -1).repeat(y.shape[0], 1)),
                dim=-1)
            q_i, context = self._noise_distr.forward(net_input)

            if sample:
                with torch.no_grad():
                    y[:, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()

            log_q_y += self._noise_distr.inner_log_prob(q_i, y[:, i].unsqueeze(dim=-1)).squeeze()
            log_p_tilde_y += self._model_log_probs(y[:, i].reshape(-1, 1), context.reshape(-1, self.num_context_units))

        if return_samples:
            return log_p_tilde_y, log_q_y, y
        else:
            return log_p_tilde_y, log_q_y

    def set_num_proposal_samples_validation(self,
                                            num_proposal_samples_validation=int(
                                                1e2)):
        assert not self.training, 'Model must be in eval mode.'
        self.num_neg_samples_validation = int(num_proposal_samples_validation)

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def set_training(self, training: bool):
        self.training = training

    def get_proposal(self):
        return self._noise_distr

