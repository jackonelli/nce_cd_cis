# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemCisJointCrit(AemIsJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y: tuple, y_samples: tuple):

        y, context_y, log_q_y = y
        y_samples, context_y_samples, log_q_y_samples = y_samples

        log_p_tilde_y, log_p_tilde_y_samples = self._model_log_probs(y, y_samples, context_y, context_y_samples,
                                                                     self._num_neg)

        # calculate log normalizer
        log_w_tilde_y_s = torch.cat(((log_p_tilde_y - log_q_y.detach()).reshape(-1, 1),
                                     (log_p_tilde_y_samples - log_q_y_samples.detach()).reshape(-1, self._num_neg)),
                                    dim=1)
        assert log_w_tilde_y_s.shape == (y.shape[0], 1 + self._num_neg)
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1]))

        # calculate normalized density
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss


