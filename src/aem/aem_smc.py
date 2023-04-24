# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch
from torch.distributions import Categorical

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCrit(AemIsJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, _idx):
        return self.inner_crit((y,), (None,))

    def inner_crit(self, y: tuple, y_samples: tuple):

        y = y[0]
        assert y.ndim == 2
        assert y.shape[0] > 1

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros((y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units))

        # Estimate log normalizer
        log_w_tilde_y_samples, _ = self.smc(y.shape[0])
        log_normalizer = torch.sum(
            torch.logsumexp(log_w_tilde_y_samples, dim=1) - torch.log(torch.Tensor([self._num_neg])), dim=-1)

        # calculate normalized density
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss

    def smc(self, batch_size, y=None):

        if y is not None:
            assert batch_size == y.shape[0]
            num_observed = batch_size
            num_chains = self._num_neg + 1
            y_s = torch.cat((y.unsqueeze(dim=1), torch.zeros((batch_size, self._num_neg, self.dim))), dim=1)
        else:
            num_observed = 0
            num_chains = self._num_neg
            y_s = torch.zeros((batch_size, num_chains, self.dim))

        log_w_tilde_y_s = torch.zeros((batch_size, num_chains, self.dim))
        log_q_y_s = torch.zeros((batch_size, num_chains, self.dim))

        # First dim
        # Propagate
        log_q_y_s[:, :, 0], context, y_s = self._proposal_log_probs(y_s.reshape(-1, 1), 0, num_observed=num_observed)

        # Reweight
        log_p_tilde_y_s = self._model_log_probs(y_s[:, :, 0].reshape(-1, 1), context.reshape(-1, self.num_context_units))
        log_w_tilde_y_s[:, :, 0] = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_chains, self.dim)

        # Dim 2 to D
        for i in range(1, self.dim):

            # Resample
            with torch.no_grad(): # TODO: eller?
                #ancestor_inds = Categorical(logits=log_w_tilde_y_s[:, :, i-1]).sample(sample_shape=torch.Size(self._num_neg + 1,))
                ancestor_inds = Categorical(logits=log_w_tilde_y_s[:, :, i].transpose(1, 2)
                                            ).sample(sample_shape=num_chains,).transpose(2, 1)

            #y_s[:, :, i-1] = torch.gather(y_s[:, :, i-1], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i)) # TODO: check this with simple example SKA JAG BACKPROPPA HELA KEDJAN?
            #log_w_tilde_y_s[:, :, i-1] = torch.gather(log_w_tilde_y_s[:, :, i-1], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i)) # TODO: check this with simple example SKA JAG BACKPROPPA HELA KEDJAN?
            y_s[:, :, :i] = torch.gather(y_s[:, :, :i], dim=1, index=ancestor_inds)
            log_w_tilde_y_s[:, :, :i] = torch.gather(log_w_tilde_y_s[:, :, :i], dim=1, index=ancestor_inds)
            log_q_y_s[:, :, :i] = torch.gather(log_q_y_s[:, :, :i], dim=1, index=ancestor_inds)

            # Propagate
            log_q_y_s[:, :, i], context, y_s = self._proposal_log_probs(y_s.reshape(-1, 1), i, num_observed=num_observed)

            # Reweight
            log_p_tilde_y_s = self._model_log_probs(y_s[:, :, i].reshape(-1, 1), context.reshape(-1, self.num_context_units))
            log_w_tilde_y_s[:, :, i] = (log_p_tilde_y_s - log_q_y_s[:, :, i].detach()).reshape(-1, num_chains, self.dim) # TODO: övriga termer tar ut varandra?

        return log_w_tilde_y_s, log_q_y_s

    def _proposal_log_probs(self, y, dim: int, num_observed: int = 0):

        net_input = torch.cat(
            (y * self.mask[dim, :], self.mask[dim, :].reshape(1, -1).repeat(y.shape[0], 1)),
            dim=-1)
        q_i, context = self._noise_distr.forward(net_input)

        y[num_observed:, dim] = self._noise_distr.inner_sample(q_i, torch.Size((1,)))[num_observed:]
        log_q_y_s = self._noise_distr.inner_log_prob(q_i, y[:, dim].reshape(-1, self.dim))

        return log_q_y_s, context, y

    def _model_log_probs(self, y, context):

        energy_net_inputs = torch.cat((y, context), dim=-1)

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

        return energy_net_outputs

    def log_prob(self, y):

        y = y[0]
        assert y.ndim == 2
        assert y.shape[0] > 1

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros(
            (y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units))

        # Estimate log normalizer
        log_w_tilde_y_samples, _ = self.smc(y.shape[0])

        log_normalizer = torch.sum(
            torch.logsumexp(log_w_tilde_y_samples, dim=1) - torch.log(torch.Tensor([self._num_neg + 1])), dim=-1)

        log_prob_p = torch.sum(log_p_tilde_y, dim=-1) - log_normalizer
        log_prob_q = torch.sum(log_q_y, dim=-1)

        return log_prob_p, log_prob_q



