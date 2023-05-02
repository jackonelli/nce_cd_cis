# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemCisJointCrit(AemIsJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y, y_samples=None):

        # Calculate (unnormalized) densities
        log_p_tilde_y, log_q_y, _, _, log_normalizer, y_samples = self._log_probs(y, self._num_neg)

        # Calculate loss
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_samples

    def _log_probs(self, y, num_samples):
        # Calculate (unnormalized) densities
        log_q_y, log_q_y_samples, context, y_samples = self._proposal_log_probs(y, num_samples=num_samples)

        log_p_tilde_y_s = torch.sum(self._model_log_probs(torch.cat((y, y_samples.detach()), dim=0).reshape(-1, 1),
                                                          context.reshape(-1, self.num_context_units)).reshape(-1, self.dim), dim=-1)

        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_y_s[:y.shape[0]], log_p_tilde_y_s[y.shape[0]:]

        # Calculate log normalizer
        log_w_tilde_y_s = torch.cat(((log_p_tilde_y - log_q_y.detach()).reshape(-1, 1),
                                     (log_p_tilde_y_samples - log_q_y_samples.detach()).reshape(-1, num_samples)),
                                    dim=1)
        assert log_w_tilde_y_s.shape == (y.shape[0], num_samples + 1)
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples + 1]))

        return log_p_tilde_y, log_q_y, log_q_y_samples, log_w_tilde_y_s, log_normalizer, y_samples

    def log_prob(self, y):

        # Calculate (unnormalized) densities
        log_p_tilde_y, log_q_y, _, _, log_normalizer, y_samples = self._log_probs(y, self.num_neg_samples_validation)

        # Calculate/estimate normalized density
        log_prob_p = log_p_tilde_y - log_normalizer
        log_prob_q = log_q_y

        return log_prob_p, log_prob_q


