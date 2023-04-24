# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.models.aem.made_joint_z import get_autoregressive_mask


class AemIsJointCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int, num_neg_samples_validation: int=1e2,
                 alpha: float = 1.0):

        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.alpha = alpha
        self.dim = self._noise_distr.num_features
        self.num_context_units = self._noise_distr.num_context_units
        self.mask = get_autoregressive_mask(self.dim)
        self.num_neg_samples_validation = num_neg_samples_validation
        self.training = True

    def crit(self, y, _idx):

        log_q_y, log_q_y_samples, context_y, context_y_samples, y_samples = self._proposal_log_probs(y, num_samples=self._num_neg)

        return self.inner_crit((y, context_y, log_q_y), (y_samples, context_y_samples, log_q_y_samples))

    def inner_crit(self, y: tuple, y_samples: tuple):

        y, context_y, log_q_y = y
        y_samples, context_y_samples, log_q_y_samples = y_samples

        log_p_tilde_y, log_p_tilde_y_samples = self._model_log_probs(y, y_samples, context_y, context_y_samples, self._num_neg)

        # calculate log normalizer
        log_normalizer = torch.logsumexp((log_p_tilde_y_samples - log_q_y_samples.detach()).reshape(-1, self._num_neg), # stop gradient
                                         dim=1) - torch.log(torch.Tensor([self._num_neg]))

        # calculate normalized density
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss

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

    def _proposal_log_probs(self, y, num_samples: int):

        y_samples = torch.zeros((y.shape[0] * num_samples, self.dim))
        context = torch.zeros((y.shape[0] * (num_samples + 1), self.dim, self.num_context_units))
        log_q_y_s = torch.zeros((y.shape[0] * (num_samples + 1), self.dim))

        y_s = torch.cat((y * self.mask[0, :], y_samples))
        for i in range(self.dim):
            net_input = torch.cat((y_s, self.mask[i, :].reshape(1, -1).repeat(y_s.shape[0], 1)), dim=-1)
            q_i, context[:, i, :] = self._noise_distr.forward(net_input)

            with torch.no_grad():
                y_samples[:, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()[y.shape[0]:]

            y_s = torch.cat((y * self.mask[i + 1, :], y_samples))
            log_q_y_s[:, i] = self._noise_distr.inner_log_prob(q_i, y_s[:, i].unsqueeze(dim=-1)).squeeze()

        log_q_y_s = log_q_y_s.sum(dim=-1)
        context_y, context_y_samples = context[:y.shape[0], ::], context[y.shape[0]:, ::]
        log_q_y, log_q_y_samples = log_q_y_s[:y.shape[0]], log_q_y_s[y.shape[0]:]

        return log_q_y, log_q_y_samples, context_y, context_y_samples, y_samples

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

    def _model_log_probs(self, y, y_samples, context_y, context_y_samples, num_proposal_samples):

        y_s = torch.cat((y, y_samples.detach()), dim=0).reshape(-1, 1)
        context_params_tiled = torch.cat((context_y, context_y_samples)).reshape(-1, self.num_context_units)

        del context_y, context_y_samples  # free GPU memory

        energy_net_inputs = torch.cat(
            (y_s, context_params_tiled),
            dim=-1
        )

        del y_s, context_params_tiled  # free GPU memory

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
        energy_net_outputs = energy_net_outputs.reshape(-1, self.dim)
        assert energy_net_outputs.shape[0] == (y.shape[0] + y_samples.shape[0])

        log_p_tilde_y = torch.sum(energy_net_outputs[:y.shape[0], :], dim=-1)
        log_p_tilde_y_samples = torch.sum(energy_net_outputs[y.shape[0]:, :], dim=-1)

        return log_p_tilde_y, log_p_tilde_y_samples

    def log_prob(self, y):

        log_q_y, log_q_y_samples, context_y, context_y_samples, y_samples = self._proposal_log_probs(y, num_samples=self._num_neg)

        log_p_tilde_y, log_p_tilde_y_samples = self._model_log_probs(y, y_samples, context_y, context_y_samples,
                                                                     self.num_neg_samples_validation)

        # Calculate joint distr. over all dimensions
        log_p_tilde_y, log_p_tilde_y_samples = torch.sum(log_p_tilde_y, dim=-1), torch.sum(log_p_tilde_y_samples, dim=-1)
        log_q_y, log_q_y_samples = torch.sum(log_q_y, dim=-1), torch.sum(log_q_y_samples, dim=-1)

        # calculate log normalizer
        log_normalizer = torch.logsumexp((log_p_tilde_y_samples - log_q_y_samples.detach()).reshape(-1, self.num_neg_samples_validation),  # stop gradient
                                         dim=1) - torch.log(torch.Tensor([self.num_neg_samples_validation]))

        # calculate normalized density
        log_prob_p = log_p_tilde_y - log_normalizer
        log_prob_q = log_q_y

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

    def part_fn(self, y, y_samples) -> Tensor:
        return 0

    def set_num_proposal_samples_validation(self,
                                            num_proposal_samples_validation=int(
                                                1e2)):
        assert not self.training, 'Model must be in eval mode.'
        self.num_neg_samples_validation = num_proposal_samples_validation

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def set_training(self, training: bool):
        self.training = training

    def get_proposal(self):
        return self._noise_distr

