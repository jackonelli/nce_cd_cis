"""Noise Contrastive Estimation (NCE) for ACE"""
import torch

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel

from src.ace.ace_is import AceIsCrit


class AceCisAdaCrit(AceIsCrit):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0,
        mask_generator=None,
        device=torch.device("cpu"),
        batch_size=None
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, alpha, energy_reg, mask_generator, device)

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        # Note that we calculate the criterion and not the gradient directly
        # Note: y, y_samples are tuples

        y_u, observed_mask, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples, _ = \
            self._log_probs(y_u, y_samples, observed_mask, context, q)
        assert log_q_y.shape == (y_u.shape[0], y_u.shape[-1])

        log_w_tilde_y = (log_p_tilde_y - log_q_y.detach().clone()) * (1 - observed_mask)
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)

        log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples), dim=1)
        assert log_w_tilde_y_s.shape == (y_u.shape[0], 1 + self._num_neg, y_u.shape[-1])

        log_z = (torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.tensor(self._num_neg + 1))) * (
                    1 - observed_mask)  # TODO log(J+1) is part of CIS estimate (but not RNCE?)

        log_p_y = log_p_tilde_y - log_z
        assert log_p_y.shape == (y_u.shape[0], y_u.shape[-1])

        p_loss = - self.alpha * torch.mean(torch.sum(log_p_y, dim=-1))

        weights = torch.nn.Softmax(dim=1)(log_w_tilde_y_s).detach().clone()
        log_q_y_s = torch.cat((log_q_y.unsqueeze(dim=1), log_q_y_samples), dim=1) * (1 - observed_mask).unsqueeze(dim=1)
        assert log_q_y_s.shape == (y_u.shape[0], 1 + self._num_neg, y_u.shape[-1])

        q_loss = - torch.mean(torch.sum(weights * log_q_y_s, dim=(1, -1)))
        #q_loss = - torch.mean(torch.sum(weights[:, 0, :] * log_q_y + torch.sum(weights[:, 1:, :] * log_q_y_samples, dim=1), dim=-1))

        if self.energy_reg != 0.0:
            p_loss += self.energy_reg * torch.nn.MSELoss()(log_p_y, log_q_y.detach().clone())

        loss = q_loss + p_loss

        return loss, p_loss, q_loss
