# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

from typing import Optional
import torch
from torch import Tensor
from torch import distributions

from src.part_fn_base import PartFnEstimator
from src.noise_distr.aem_proposal import AemProposal


class AemIsCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr: AemProposal, num_neg_samples: int, num_neg_samples_validation: int=1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.alpha = alpha
        self.dim = self._noise_distr.autoregressive_net.input_dim
        self.num_neg_samples_validation = num_neg_samples_validation
        self.training = True

    def crit(self, y, conditional_inputs=None):

        q, context = self._noise_distr.forward(y, conditional_inputs)

        if self._noise_distr.Component is not None:
            y_samples = self.inner_sample_noise(q, num_samples=self._num_neg) #.transpose(1, 2)
            assert y_samples.shape == (y.shape[0], self._num_neg, y.shape[1])
        else:
            y_samples = self.inner_sample_noise(q, num_samples=self._num_neg * y.shape[0] * y.shape[1]
                                                ).reshape(y.shape[0], self._num_neg, y.shape[1])  #(y.shape[0], y.shape[1], self._num_neg)

        return self.inner_crit((y, context), (y_samples, q))

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        y, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples =\
            self._log_probs(y, y_samples, context, q, self._num_neg)

        # calculate log normalizer
        log_normalizer = torch.logsumexp(log_p_tilde_y_samples - log_q_y_samples.detach(),  # stop gradient
                                         dim=1) - torch.log(torch.Tensor([self._num_neg]))

        # calculate normalized density
        p_loss = - torch.mean(torch.sum(
            log_p_tilde_y - log_normalizer,
            dim=-1
        ))
        q_loss = - torch.mean(torch.sum(
            log_q_y,
            dim=-1
        ))

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

    def _log_probs(self, y, y_samples, context, q, num_proposal_samples):
        # evaluate data and proposal samples under proposal
        log_q_y_samples = self._noise_distr.inner_log_prob(q, y_samples)  # [B, J, D] ([B, D, S])
        log_q_y = self._noise_distr.inner_log_prob(q, y.unsqueeze(dim=1)).reshape(-1, self.dim)

        # energy net
        inputs_cat_samples = torch.cat(
            (y.unsqueeze(dim=1), y_samples.detach()),  # stop gradient y[..., None]
            dim=1
        )

        inputs_cat_samples = inputs_cat_samples.reshape(-1, 1)
        context_params = context.unsqueeze(dim=1)
        context_params_tiled = context_params.repeat(
            1, num_proposal_samples + 1, 1, 1
        )
        del context_params  # free GPU memory
        context_params_tiled = context_params_tiled.reshape(-1, self._noise_distr.num_context_units)


        energy_net_inputs = torch.cat(
            (inputs_cat_samples, context_params_tiled),
            dim=-1
        )
        del inputs_cat_samples, context_params_tiled  # free GPU memory

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

        energy_net_outputs = energy_net_outputs.reshape(-1, 1 + num_proposal_samples, self.dim)

        # unnormalized log densities given by energy net
        log_p_tilde_y = energy_net_outputs[:, 0, :]
        log_p_tilde_y_samples = energy_net_outputs[:, 1:, :]

        return log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples

    def log_prob(self, y, conditional_inputs=None):

        q, context = self._noise_distr.forward(y, conditional_inputs)

        if self._noise_distr.Component is not None:
            y_samples = self.inner_sample_noise(q, num_samples=self.num_neg_samples_validation)
        else:
            y_samples = self.inner_sample_noise(q, num_samples=self.num_neg_samples_validation * y.shape[0] * y.shape[1]
                                                ).reshape(y.shape[0], self.num_neg_samples_validation, y.shape[1])

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples = \
            self._log_probs(y, y_samples, context, q, self.num_neg_samples_validation)

        # calculate log normalizer
        log_normalizer = torch.logsumexp(log_p_tilde_y_samples - log_q_y_samples.detach(),  # stop gradient
                                         dim=1) - torch.log(torch.Tensor([self.num_neg_samples_validation]))

        # calculate normalized density
        log_prob_p = torch.sum(log_p_tilde_y - log_normalizer, dim=-1)
        log_prob_q = torch.sum(log_q_y, dim=-1)

        return log_prob_p, log_prob_q

    def set_num_proposal_samples_validation(self,
                                            num_proposal_samples_validation=int(
                                                        1e2)):
        assert not self.training, 'Model must be in eval mode.'
        self.num_neg_samples_validation = num_proposal_samples_validation

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def set_training(self, training: bool):
        self.training = training

    def sample(self, batch_size=1000):
        samples = torch.zeros(batch_size, self.dim)

        num_proposal_samples = 100
        for dim in range(self.dim):

            q, context = self._noise_distr.forward_along_dim(samples, dim)

            if self._noise_distr.Component is not None:
                y_samples = self.inner_sample_noise(q, num_samples=num_proposal_samples)
            else:
                y_samples = self.inner_sample_noise(q, num_samples=num_proposal_samples * batch_size
                                                    ).reshape(batch_size, num_proposal_samples, 1)

            # reshape for log prob calculation
            proposal_log_density = self._noise_distr.inner_log_prob(q, y_samples)
            proposal_log_density = proposal_log_density.reshape(batch_size, num_proposal_samples)

            # reshape again for input to energy net
            y_samples = y_samples.reshape(-1, 1)
            energy_net_inputs = torch.cat(
                (y_samples,
                 context.repeat(1, num_proposal_samples, 1).reshape(
                     batch_size * num_proposal_samples, -1)),
                dim=-1
            )
            unnormalized_log_density = self._unnorm_distr(energy_net_inputs)
            unnormalized_log_density = unnormalized_log_density.reshape(batch_size,
                                                                        num_proposal_samples)
            logits = unnormalized_log_density - proposal_log_density
            resampling_distribution = distributions.Categorical(logits=logits)
            selected_indices = resampling_distribution.sample((1,)).reshape(-1)
            y_samples = y_samples.reshape(batch_size, num_proposal_samples)
            selected_points = y_samples[range(batch_size), selected_indices]

            samples[:, dim] += selected_points.detach()

        return samples

    def inner_sample_noise(self, q, num_samples: int):
        with torch.no_grad():
            y_samples = self._noise_distr.inner_sample(q, torch.Size((num_samples,)))

        return y_samples

    def part_fn(self, y, y_samples) -> Tensor:
        return 0

    def get_proposal(self):
        return self._noise_distr

