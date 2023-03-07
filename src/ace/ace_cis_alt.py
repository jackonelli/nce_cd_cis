"""Noise Contrastive Estimation (NCE) for ACE (calculating loss with persistent y)"""
import torch

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel

from src.ace.ace_is import AceIsCrit


class AceCisAltCrit(AceIsCrit):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, alpha, energy_reg)

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None, return_weights=False):

        # Note that we calculate the criterion and not the gradient directly
        # Note: y, y_samples are tuples
        y_u_ext, observed_mask, context = y
        y_samples, q = y_samples

        # Group persistent y with samples (they are all used in weight calculations)
        # If y_base is None, we do some extra (unnecessary) calculations
        y_samples = torch.cat((y_u_ext.unsqueeze(dim=1), y_samples), dim=1)  # Should not matter that we use only unobserved for y

        if y_base is None:
            y_u = y_u_ext.clone()
        else:
            y_u = y_base * (1 - observed_mask)

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples = \
            self._log_probs(y_u, y_samples, observed_mask, context, q)

        assert log_p_tilde_y_samples.shape == (y_u.shape[0], 1 + self._num_neg, y_u.shape[-1])
        assert log_q_y.shape == (y_u.shape[0], y_u.shape[-1])

        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)

        weights = torch.nn.Softmax(dim=1)(log_w_tilde_y_samples).detach().clone()
        assert weights.shape == (y_u.shape[0], 1 + self._num_neg, y_u.shape[-1])

        log_p_y_semi_grad = log_p_tilde_y - (torch.sum(weights * log_p_tilde_y_samples, dim=1) * (1 - observed_mask))
        assert log_p_y_semi_grad.shape == (y_u.shape[0], y_u.shape[-1])

        p_loss = - self.alpha * torch.mean(torch.sum(log_p_y_semi_grad, dim=-1))

        # We do not use persistence for proposal loss
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        if self.energy_reg != 0.0:
            # We also use the true data points in regularisation (note that we could use the kernel here also)
            log_w_tilde_y = (log_p_tilde_y - log_q_y.detach().clone()) * (1 - observed_mask)
            log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples[:, 1:, :]), dim=1)
            log_z = (torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.tensor(self._num_neg + 1))) * (
                    1 - observed_mask)
            p_loss += self.energy_reg * torch.nn.MSELoss()(log_p_tilde_y - log_z, log_q_y.detach().clone())

        loss = q_loss + p_loss

        if return_weights:
            return loss, p_loss, q_loss, log_w_tilde_y_samples

        else:
            return loss, p_loss, q_loss
