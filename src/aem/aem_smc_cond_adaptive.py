# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch
from torch.distributions import Categorical

from src.aem.aem_smc_adaptive import AemSmcAdaCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCondAdaCrit(AemSmcAdaCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y, y_samples=None):

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros(
            (y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units)).reshape(-1,
                                                                                                                     self.dim)

        # Estimate log normalizer
        log_normalizer, log_q, y_s = self.smc(y.shape[0], self._num_neg, y=y)

        # Calculate loss
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(log_q)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_s

    def inner_smc(self, batch_size, num_samples, y):

        assert batch_size == y.shape[0]
        y_samples = torch.zeros(batch_size, num_samples, self.dim)
        y_s = torch.cat((y.unsqueeze(dim=1), y_samples), dim=1)
        log_q_y_s = torch.zeros((batch_size, num_samples + 1, self.dim))

        # First dim
        # Propagate
        logq, context, y_s = self._proposal_log_probs(torch.cat((y_s[:, 0, :],
                                                                      y_s[:, 1:, :].reshape(-1, self.dim))), 0, num_observed=batch_size)
        log_q_y_s[:, :, 0] = torch.cat((logq[:y.shape[0]].reshape(-1, 1),
                                        logq[y.shape[0]:].reshape(-1, num_samples)), dim=1)
        del logq

        y_s = torch.cat(
            (y_s[:y.shape[0], :].unsqueeze(dim=1), y_s[y.shape[0]:, :].reshape(-1, num_samples, self.dim)), dim=1)
        assert torch.allclose(y_s[:, 0, :], y)

        # Reweight
        log_p_tilde_y_s = self._model_log_probs(torch.cat((y_s[:, 0, 0].reshape(-1, 1), y_s[:, 1:, 0].reshape(-1, 1))),
                                                context)
        del context

        log_w_tilde_y_s = torch.cat(((log_p_tilde_y_s[:y.shape[0]].reshape(-1, 1) - log_q_y_s[:, 0, 0].detach().unsqueeze(dim=1)),
                                     (log_p_tilde_y_s[y.shape[0]:].reshape(-1, num_samples) - log_q_y_s[:, 1:, 0].detach())),
                                    dim=1)
        print((log_p_tilde_y_s[y.shape[0]:].reshape(-1, num_samples) - log_q_y_s[:, 1:, 0].detach()).shape)

        del log_p_tilde_y_s

        # Dim 2 to D
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples + 1]))
        for i in range(1, self.dim):

            # Resample
            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
            
            ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach()

            resampling_inds = ess < ((num_samples + 1) / 2)
            log_weight_factor = torch.zeros(log_w_tilde_y_s.shape)

            if resampling_inds.sum() > 0:
                with torch.no_grad():
                    ancestor_inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(sample_shape=torch.Size((num_samples,))).transpose(0, 1)
                    # Do not resample y
                    y_s[resampling_inds, 1:, :i] = torch.gather(y_s[resampling_inds, :, :i], dim=1,
                                                                index=ancestor_inds[:, :, None].repeat(1, 1, i))
                                                                
                    log_q_y_s[resampling_inds, 1:, :i] = torch.gather(log_q_y_s[resampling_inds, :, :i], dim=1,
                                                                      index=ancestor_inds[:, :, None].repeat(1, 1, i))
                    assert torch.allclose(y_s[:, 0, :], y)

            #if resampling_inds.sum() < batch_size:
            log_weight_factor[~resampling_inds, :] = log_w_y_s[~resampling_inds, :] + torch.log(torch.Tensor([num_samples + 1]))
            del log_w_y_s, ess, resampling_inds

            # Propagate
            logq, context, y_s = self._proposal_log_probs(torch.cat((y_s[:, 0, :],
                                                                      y_s[:, 1:, :].reshape(-1, self.dim))), i, num_observed=y.shape[0])

            y_s = torch.cat(
                (y_s[:y.shape[0], :].unsqueeze(dim=1), y_s[y.shape[0]:, :].reshape(-1, num_samples, self.dim)), dim=1)
            assert torch.allclose(y_s[:, 0, :], y)

            log_q_y_s[:, :, i] = torch.cat((logq[:y.shape[0]].reshape(-1, 1),
                                            logq[y.shape[0]:].reshape(-1, num_samples)), dim=1)
            del logq

            # Reweight
            log_p_tilde_y_s = self._model_log_probs(torch.cat((y_s[:, 0, i].reshape(-1, 1), y_s[:, 1:, i].reshape(-1, 1))),
                                                    context)
            del context
            log_w_tilde_y_s = torch.cat(((log_p_tilde_y_s[:y.shape[0]].reshape(-1, 1) - log_q_y_s[:, 0, i].detach().unsqueeze(dim=1)),
                                         (log_p_tilde_y_s[y.shape[0]:].reshape(-1, num_samples) - log_q_y_s[:, 1:,
                                                                                                  i].detach())),
                                        dim=1)
            del log_p_tilde_y_s

            log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples + 1]))
            log_q = torch.sum(torch.nn.Softmax(dim=-1)(log_w_tilde_y_s.detach()) * torch.sum(log_q_y_s, dim=-1),
                              dim=-1)  # torch.exp(log_normalizer) *

        return log_normalizer, log_q, y_s, log_w_tilde_y_s

    def log_prob(self, y):

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros(
            (y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units)).reshape(-1,
                                                                                                                     self.dim)

        # Estimate log normalizer
        log_normalizer, _ = self.smc(y.shape[0], self.num_neg_samples_validation, y=y)

        # Calculate/estimate normalized densities
        log_prob_p = torch.sum(log_p_tilde_y, dim=-1) - log_normalizer
        log_prob_q = torch.sum(log_q_y, dim=-1)

        return log_prob_p, log_prob_q
