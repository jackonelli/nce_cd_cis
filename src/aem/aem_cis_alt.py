"""Noise Contrastive Estimation (NCE) for ACE (calculating loss with persistent y)"""
from typing import Optional
import torch
from torch import Tensor

from src.noise_distr.aem_proposal import AemProposal
from src.aem.aem_cis_joint_z import AemCisJointCrit


class AceCisJointAltCrit(AemCisJointCrit):
    def __init__(self, unnorm_distr, noise_distr: AemProposal, num_neg_samples: int,
                 num_neg_samples_validation: int = 1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, conditional_inputs=None, _idx: Optional[Tensor]=None):

        q, context = self._noise_distr.forward(y, conditional_inputs)

        if self._noise_distr.Component is not None:
            y_samples = self.inner_sample_noise(q, num_samples=self._num_neg) #.transpose(1, 2)
        else:
            y_samples = self.inner_sample_noise(q, num_samples=self._num_neg * y.shape[0] * y.shape[1]
                                                ).reshape(y.shape[0], self._num_neg, y.shape[1])  #(y.shape[0], y.shape[1], self._num_neg)

        loss, p_loss, q_loss, _ = self.inner_crit((y, context), (y_samples, q))

        return loss, p_loss, q_loss

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        y, context = y
        y_samples, q = y_samples

        # Group persistent y with samples (they are all used in weight calculations)
        # If y_base is None, we do some extra (unnecessary) calculations
        y_samples = torch.cat((y.unsqueeze(dim=1), y_samples), dim=1)

        if y_base is None:
            y_base = y.clone()

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples = \
            self._log_probs(y_base, y_samples, context, q, self._num_neg + 1)

        assert log_p_tilde_y_samples.shape == (y.shape[0], 1 + self._num_neg, y.shape[-1])
        assert log_q_y.shape == y_base.shape

        # Calculate joint distr. over all dimensions
        log_p_tilde_y, log_p_tilde_y_samples = torch.sum(log_p_tilde_y, dim=-1), torch.sum(log_p_tilde_y_samples,
                                                                                           dim=-1)
        log_q_y, log_q_y_samples = torch.sum(log_q_y, dim=-1), torch.sum(log_q_y_samples, dim=-1)

        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach())
        weights = torch.nn.Softmax(dim=1)(log_w_tilde_y_samples).detach()
        assert weights.shape == (y.shape[0], 1 + self._num_neg)

        # This is a proxy to calculating the gradient directly
        log_p_y_semi_grad = log_p_tilde_y - torch.sum(weights * log_p_tilde_y_samples, dim=1)
        assert log_p_y_semi_grad.shape == (y.shape[0],)

        p_loss = - torch.mean(log_p_y_semi_grad)

        # We do not use persistence for proposal loss
        q_loss = - torch.mean(log_q_y)

        # Note: gradient w.r.t. context might be different
        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss, log_w_tilde_y_samples.detach()
