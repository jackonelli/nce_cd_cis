"""Noise Contrastive Estimation (NCE) ranking partition functions with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import concat_samples

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel
from src.experiments.ace_exp_utils import UniformMaskGenerator, BernoulliMaskGenerator

from src.nce.ace_is import AceIsCrit


class AceCisCrit(AceIsCrit):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, alpha, energy_reg)


    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        # Note that we calculate the criterion and not the gradient directly
        # Note: y, y_samples are tuples
        # TODO: For persistance (y_base is not None), it might be easier to ass y_base also to ys

        y_o, y_u, observed_mask, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples = \
            self._log_probs(y_o, y_u, y_samples, observed_mask, context, q)

        log_w_tilde_y = (log_p_tilde_y - log_q_y.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)
        log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples), dim=1)
        log_z = (torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.tensor(self._num_neg))) * (
                    1 - observed_mask)

        log_p_y = log_p_tilde_y - log_z
        #is_weights = torch.nn.Softmax(dim=-1)(log_w_tilde_y_samples)
        #energy_mean = torch.sum(is_weights * y_samples_u) * (1 - observed_mask)

        p_loss = - self.alpha * torch.mean(torch.sum(log_p_y, dim=-1))
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        if self.energy_reg != 0.0:
            p_loss += self.energy_reg * torch.nn.MSELoss()(log_p_y, log_q_y.detach().clone())

        loss = q_loss + p_loss

        return loss, p_loss, q_loss

