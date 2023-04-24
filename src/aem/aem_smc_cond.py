# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch
from torch.distributions import Categorical

from src.aem.aem_smc import AemSmcCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCondCrit(AemSmcCrit):
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
        log_w_tilde_y_s, _ = self.smc(y.shape[0], y=y)
        log_normalizer = torch.sum(
            torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1])), dim=-1)

        # calculate normalized density
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss

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
        log_w_tilde_y_s, _ = self.smc(y.shape[0], y=y)
        log_normalizer = torch.sum(
            torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1])), dim=-1)

        log_prob_p = torch.sum(log_p_tilde_y, dim=-1) - log_normalizer
        log_prob_q = torch.sum(log_q_y, dim=-1)

        return log_prob_p, log_prob_q



