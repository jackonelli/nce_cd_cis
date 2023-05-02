"""Persistent Noise Contrastive Estimation (NCE) for the ranking criterion for ACE

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.
"""

from typing import Optional
import torch
from torch import Tensor
from src.part_fn_utils import concat_samples
from torch.distributions import Categorical

from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.aem.aem_cis_alt import AceCisJointAltCrit


class AemCisJointPersCrit(AceCisJointAltCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int = 1e2, batch_size: int = 512,  alpha: float = 1.0):

        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

        self.batch_size = batch_size
        self._persistent_y = None

    def crit(self, y: Tensor, idx: Optional[int]=None):

        if self.training:
            y_p = self.persistent_y(y, idx).to(y.device)
            loss, p_loss, q_loss, y_samples, log_w_tilde = self.inner_pers_crit(y, y_p)
            self._update_persistent_y(log_w_tilde.detach(), y_samples[:, 0, :], y_samples[:, 1:, :], idx)
        else:
            loss, p_loss, q_loss = self.inner_crit(y)

        return loss, p_loss, q_loss

    def persistent_y(self, actual_y: Tensor, idx: int):
        """Get persistent y

        Access the last selected y or return the y sampled from p_d
        if no last selected y exists
        """
        #per_y = torch.empty(actual_y.size())
        #for n, per_n in enumerate(idx):
        #    per_n = per_n.item()
        #    per_y[n, :] = (
        #        self._persistent_y[per_n]
        #        if self._persistent_y.get(per_n) is not None
        #        else actual_y[n, :]
        #    )

        if (self._persistent_y is None) or (idx is None):
            per_y = actual_y.clone()
        else:
            per_y = self._persistent_y[:idx, :].clone()

        return per_y

    def _update_persistent_y(self, log_w_unnorm, y, y_samples, idx):
        """Sample new persistent y"""

        ys = concat_samples(y, y_samples)
        with torch.no_grad():
            sampled_idx = Categorical(logits=log_w_unnorm).sample()
            y_p = torch.gather(ys, dim=1, index=sampled_idx[:, None, None].repeat(1, 1, y.shape[-1])).squeeze(dim=1)

            if self._persistent_y is None or idx is None:
                assert y.shape[0] == self.batch_size
                self._persistent_y = y_p.clone()
                #assert self._persistent_y.shape == y.shape
            else:
                self._persistent_y[:idx, :] = y_p.clone()

        #assert y_p.shape == y.shape
            #for n, _ in enumerate(ys):
                #sampled_idx = Categorical(logits=log_w_unnorm[n, :, :].transpose(0, 1)).sample()
            #    self._persistent_y[idx[n].item()] = y_p[n, :]
            #   y_p = torch.gather(ys, dim=1, index=sampled_idx.unsqueeze(dim=1)).squeeze(dim=1)

