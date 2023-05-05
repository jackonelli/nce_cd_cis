# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch
from torch.distributions import Categorical

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCrit(AemIsJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)
        #self.counter = 0

    def crit(self, y, _idx):
        loss, p_loss, q_loss, y_s = self.inner_crit(y)

        return loss, p_loss, q_loss

    def inner_crit(self, y, y_samples=None):

        #if self.training:
        #    self.counter += 1

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros((y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        # TODO: REMOVE DETACH!
        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.detach().reshape(-1, self.num_context_units)).reshape(-1, self.dim)

        # Estimate log normalizer
        log_normalizer, y_s = self.smc(y.shape[0])


        # Calculate loss
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_s

    def smc(self, batch_size, y=None):
        # This is just a wrapper function
        log_normalizer, y_s, log_w_tilde_y_s = self.inner_smc(batch_size, y)

        return log_normalizer, y_s

    def inner_smc(self, batch_size, y):

        if y is not None:
            assert batch_size == y.shape[0]
            num_observed = batch_size
            num_chains = self._num_neg + 1
            #y_s = torch.cat((y.unsqueeze(dim=1), torch.zeros((batch_size, self._num_neg, self.dim))), dim=1)
            y_s = torch.cat((y, torch.zeros(batch_size * self._num_neg, self.dim))).reshape(-1, self._num_neg + 1, self.dim)
            assert torch.allclose(y, y_s.reshape(-1, self.dim)[:y.shape[0], :])
        else:
            num_observed = 0
            num_chains = self._num_neg
            y_s = torch.zeros((batch_size, num_chains, self.dim))

        # First dim
        # Propagate
        log_q_y_s, context, y_s = self._proposal_log_probs(y_s.reshape(-1, self.dim), 0, num_observed=num_observed)

        if num_observed > 0:
            assert torch.allclose(y, y_s[:y.shape[0], :])

        context, y_s = context.reshape(-1, num_chains, self.num_context_units), y_s.reshape(-1, num_chains, self.dim)

        # Reweight
        # TODO: REMOVE DETACH!
        log_p_tilde_y_s = self._model_log_probs(y_s[:, :, 0].reshape(-1, 1), context.detach().reshape(-1, self.num_context_units))
        del context

        log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_chains)
        del log_p_tilde_y_s, log_q_y_s

        # Dim 2 to D
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_chains]))
        for i in range(1, self.dim):

            # Resample
            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
            ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1))

            resampling_inds = ess < (num_chains / 2)
            log_weight_factor = torch.zeros(log_w_tilde_y_s.shape)

            if resampling_inds.sum() > 0:
                with torch.no_grad():
                    ancestor_inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(sample_shape=torch.Size((num_chains,))).transpose(0, 1)

                y_s[resampling_inds, :, :i] = torch.gather(y_s[resampling_inds, :, :i], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i))
                del ancestor_inds

            if resampling_inds.sum() < batch_size:
                log_weight_factor[~resampling_inds, :] = log_w_y_s[~resampling_inds, :] + torch.log(torch.Tensor([num_chains]))

            del log_w_y_s, ess, resampling_inds

            # Propagate
            log_q_y_s, context, y_s = self._proposal_log_probs(y_s.reshape(-1, self.dim), i, num_observed=num_observed)
            context, y_s = context.reshape(-1, num_chains, self.num_context_units), y_s.reshape(-1, num_chains,
                                                                                                self.dim)
            if y is not None:
                assert torch.allclose(y_s[:, 0, :], y)

            # Reweight
            # TODO: REMOVE DETACH!
            log_p_tilde_y_s = self._model_log_probs(y_s[:, :, i].reshape(-1, 1), context.detach().reshape(-1, self.num_context_units))
            del context
            log_w_tilde_y_s = log_weight_factor + (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_chains) # TODO: övriga termer tar ut varandra?
            del log_p_tilde_y_s, log_q_y_s, log_weight_factor

            log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_chains]))

        return log_normalizer, y_s, log_w_tilde_y_s

    def _proposal_log_probs(self, y, dim: int, num_observed: int = 0):
        net_input = torch.cat(
            (y * self.mask[dim, :], self.mask[dim, :].reshape(1, -1).repeat(y.shape[0], 1)),
            dim=-1)

        q_i, context = self._noise_distr.forward(net_input)

        if num_observed < y.shape[0]:
            with torch.no_grad():
                y[num_observed:, dim] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()[num_observed:]

        log_q_y_s = self._noise_distr.inner_log_prob(q_i, y[:, dim].unsqueeze(dim=-1)).squeeze()

        return log_q_y_s, context, y

    def log_prob(self, y, return_unnormalized=False):

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros(
            (y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units)).reshape(-1, self.dim)

        # Estimate log normalizer
        log_normalizer, _ = self.smc(y.shape[0])

        # Calculate/estimate normalized densities
        log_prob_p = torch.sum(log_p_tilde_y, dim=-1) - log_normalizer
        log_prob_q = torch.sum(log_q_y, dim=-1)

        if return_unnormalized:
            return log_prob_p, log_prob_q, log_p_tilde_y
        else:
            return log_prob_p, log_prob_q



