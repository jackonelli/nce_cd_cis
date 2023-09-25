"""Noise Contrastive Estimation (NCE) for ACE (calculating loss with persistent y)"""
from typing import Optional
import torch
from torch import Tensor

from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.aem.aem_cis_joint_z import AemCisJointCrit


class AceCisJointAltCrit(AemCisJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int = 1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, _idx: Optional[Tensor] = None):

        if self.training:
            loss, p_loss, q_loss, _, _ = self.inner_pers_crit(y, y)
        else:
            loss, p_loss, q_loss, _ = self.inner_crit(y)

        return loss, p_loss, q_loss

    def inner_pers_crit(self, y, y_samples):

        # Calculate (unnormalized) densities
        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples, log_w_tilde_y_samples, y_samples \
            = self._pers_log_probs(y, y_samples, self._num_neg)

        # This is a proxy to calculating the gradient directly
        log_p_y_semi_grad = log_p_tilde_y - torch.sum(torch.nn.Softmax(dim=-1)(log_w_tilde_y_samples.detach()) * log_p_tilde_y_samples, dim=1)
        assert log_p_y_semi_grad.shape == (y.shape[0],)

        # Calculate losses
        p_loss = - torch.mean(log_p_y_semi_grad)
        q_loss = - torch.mean(log_q_y)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, y_samples, log_w_tilde_y_samples

    def _pers_log_probs(self, y, y_samples, num_samples):

        log_q_y, log_q_y_samples, context, y_samples = self._pers_proposal_log_probs(y, num_samples=num_samples,
                                                                                     y_sample_base=y_samples)

        log_p_tilde_y_s = torch.sum(self._model_log_probs(torch.cat((y, y_samples.detach()), dim=0).reshape(-1, 1),
                                                          context.reshape(-1, self.num_context_units)).reshape(-1,
                                                                                                               self.dim),
                                    dim=-1)

        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_y_s[:y.shape[0]], log_p_tilde_y_s[y.shape[0]:]

        assert log_p_tilde_y_samples.shape[0] == (y.shape[0] * (self._num_neg + 1))

        log_p_tilde_y_samples = torch.cat((log_p_tilde_y_samples[:y.shape[0]].reshape(-1, 1),
                                           log_p_tilde_y_samples[y.shape[0]:].reshape(-1, num_samples)), dim=1)
        log_q_y_samples = torch.cat((log_q_y_samples[:y.shape[0]].reshape(-1, 1),
                                     log_q_y_samples[y.shape[0]:].reshape(-1, num_samples)), dim=1)

        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples).detach()
        y_samples = torch.cat((y_samples[:y.shape[0], :].reshape(-1, 1, self.dim),
                               y_samples[y.shape[0]:, :].reshape(-1, num_samples, self.dim)), dim=1)

        return log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples, log_w_tilde_y_samples, y_samples

    def _pers_proposal_log_probs(self, y, num_samples: int, y_sample_base=None):

        if y_sample_base is not None:
            y_ext = torch.cat((y, y_sample_base))
            factor = 2
        else:
            y_ext = y.clone()
            factor = 1

        y_samples = torch.zeros((y.shape[0] * num_samples, self.dim))
        context = torch.zeros((y.shape[0] * (num_samples + factor), self.dim, self.num_context_units))
        log_q_y_s = torch.zeros((y.shape[0] * (num_samples + factor), self.dim))

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