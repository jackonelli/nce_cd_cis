# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.noise_distr.aem_proposal import AemProposal


class AemCisJointCrit(AemIsJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemProposal, num_neg_samples: int, num_neg_samples_validation: int=1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        y, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples =\
            self._log_probs(y, y_samples, context, q, self._num_neg)

        # Calculate joint distr. over all dimensions
        log_p_tilde_y, log_p_tilde_y_samples = torch.sum(log_p_tilde_y, dim=-1), torch.sum(log_p_tilde_y_samples, dim=-1)
        log_q_y, log_q_y_samples = torch.sum(log_q_y, dim=-1), torch.sum(log_q_y_samples, dim=-1)

        # calculate log normalizer
        log_w_tilde_y = (log_p_tilde_y - log_q_y.detach())
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach())
        log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples), dim=1)
        assert log_w_tilde_y_s.shape == (y.shape[0], 1 + self._num_neg)
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1]))

        # calculate normalized density
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss


