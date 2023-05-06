# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_smc_cond_adaptive import AemSmcCondAdaCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCondAdaAltCrit(AemSmcCondAdaCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, _idx):

        if self.training:
            loss, p_loss, q_loss, _, _ = self.inner_pers_crit(y, y)
        else:
            loss, p_loss, q_loss = self.inner_crit(y)

        return loss, p_loss, q_loss

    def inner_pers_crit(self, y, y_samples):

        assert y.shape[0] == y_samples.shape[0]

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros((y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units)).reshape(-1, self.dim)

        # Estimate log normalizer
        log_normalizer, log_q, y_s, log_w_tilde_y_s = self.inner_smc(y_samples.shape[0], self.num_neg_samples_validation, y=y_samples)

        # Calculate loss
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(log_q)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_s, log_w_tilde_y_s



