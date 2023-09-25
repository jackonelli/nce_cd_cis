# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemCisJointAdaCrit(AemCisJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int, num_neg_samples_validation: int=1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y, y_samples=None):
        # Calculate (unnormalized) densities
        log_p_tilde_y, log_q_y, log_q_y_samples, log_w_tilde_y_s, log_normalizer, y_samples = self._log_probs(y, self._num_neg)

        # Calculate loss for model
        p_loss = - torch.mean(log_p_tilde_y - log_normalizer)

        # We also use the kernel to calculate the loss for q
        weights = torch.nn.Softmax(dim=1)(log_w_tilde_y_s).detach()
        log_q_y_s = torch.cat((log_q_y.reshape(-1, 1), log_q_y_samples.reshape(-1, self._num_neg)), dim=-1)
        assert log_q_y_s.shape == (y.shape[0], 1 + self._num_neg)
        q_loss = - torch.mean(torch.sum(weights * log_q_y_s, dim=-1))

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_samples



