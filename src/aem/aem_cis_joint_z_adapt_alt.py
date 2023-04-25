# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
from typing import Optional
import torch
from torch import Tensor

from src.aem.aem_cis_joint_z_adapt import AemCisJointAdaCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemCisJointAdaAltCrit(AemCisJointAdaCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int = 1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, _idx: Optional[Tensor] = None):

        if self.training:
            loss, p_loss, q_loss, _, _ = self.inner_pers_crit(y, y)
        else:
            loss, p_loss, q_loss = self.inner_crit(y)

        return loss, p_loss, q_loss

    def inner_pers_crit(self, y, y_samples):
        # Criteria using persistent y

        # Calculate (unnormalized) densities
        log_q_y, log_q_y_samples, context, y_samples = self._proposal_log_probs(y, num_samples=self._num_neg, y_sample_base=y_samples)

        log_p_tilde_y_s = torch.sum(self._model_log_probs(torch.cat((y, y_samples.detach()), dim=0).reshape(-1, 1),
                                                          context).reshape(-1, self.dim), dim=-1)

        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_y_s[:y.shape[0]], log_p_tilde_y_s[y.shape[0]:]

        assert log_p_tilde_y_samples.shape[0] == (y.shape[0] * (self._num_neg + 1))

        log_p_tilde_y_samples = torch.cat((log_p_tilde_y_samples[:y.shape[0]].reshape(-1, 1),
                                           log_p_tilde_y_samples[y.shape[0]:].reshape(-1, self._num_neg)), dim=1)
        log_q_y_samples = torch.cat((log_q_y_samples[:y.shape[0]].reshape(-1, 1),
                                     log_q_y_samples[y.shape[0]:].reshape(-1, self._num_neg)), dim=1)

        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples).detach()

        # This is a proxy for calculating the gradient directly
        weights = torch.nn.Softmax(dim=-1)(log_w_tilde_y_samples)
        log_p_y_semi_grad = log_p_tilde_y - torch.sum(
            weights * log_p_tilde_y_samples, dim=1)
        assert log_p_y_semi_grad.shape == (y.shape[0],)

        # Calculate loss
        p_loss = - torch.mean(log_p_y_semi_grad)
        q_loss = - torch.mean(torch.sum(weights * log_q_y_samples, dim=-1))

        # Note: gradient w.r.t. context might be different
        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_samples.detach(), log_w_tilde_y_samples.detach()

    def _proposal_log_probs(self, y, num_samples: int, y_sample_base=None):

        y_samples = torch.zeros((y.shape[0] * num_samples, self.dim))
        context = torch.zeros((y.shape[0] * (num_samples + 2), self.dim, self.num_context_units))
        log_q_y_s = torch.zeros((y.shape[0] * (num_samples + 2), self.dim))

        if y_sample_base is not None:
            y_ext = torch.cat((y, y_sample_base))
        else:
            y_ext = y.clone()

        y_s = torch.cat((y_ext * self.mask[0, :], y_samples))
        for i in range(self.dim):
            net_input = torch.cat((y_s, self.mask[i, :].reshape(1, -1).repeat(y_s.shape[0], 1)), dim=-1)
            q_i, context[:, i, :] = self._noise_distr.forward(net_input)

            with torch.no_grad():
                y_samples[:, i] = self._noise_distr.inner_sample(q_i, torch.Size((1,))).squeeze()[y_ext.shape[0]:]

            y_s = torch.cat((y_ext * self.mask[i + 1, :], y_samples))
            log_q_y_s[:, i] = self._noise_distr.inner_log_prob(q_i, y_s[:, i].unsqueeze(dim=-1)).squeeze()

        log_q_y_s = log_q_y_s.sum(dim=-1)
        log_q_y, log_q_y_samples = log_q_y_s[:y.shape[0]], log_q_y_s[y.shape[0]:]

        if y_sample_base is not None:
            y_samples = torch.cat((y_sample_base, y_samples))

        return log_q_y, log_q_y_samples, context, y_samples