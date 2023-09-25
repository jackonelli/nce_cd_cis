"""Persistent Noise Contrastive Estimation (NCE) for the ranking criterion for ACE

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.
"""

from typing import Optional
import torch
from torch import Tensor
from src.part_fn_utils import concat_samples
from torch.distributions import Categorical

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel

from src.ace.ace_cis_alt import AceCisAltCrit

class AceCisPers(AceCisAltCrit):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0,
        mask_generator=None,
        device=torch.device("cpu"),
        batch_size: int = 0,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, alpha, energy_reg, mask_generator, device)
        self._persistent_y = None
        self.batch_size = batch_size

    def crit(self, y: Tensor, idx: Optional[int]):

        # Mask input
        y_p = self.persistent_y(y, idx).to(self.device)
        y_o, y_u, observed_mask = self._mask_input(y_p)

        q, context = self._noise_distr.forward((y * observed_mask, observed_mask))
        y_samples = self.inner_sample_noise(q, num_samples=self._num_neg).detach().clone()

        if idx is None:
            loss, p_loss, q_loss = self.inner_crit((y_u, observed_mask, context), (y_samples, q))
        else:
            loss, p_loss, q_loss, log_w_tilde = self.inner_pers_crit((y_u, observed_mask, context), (y_samples, q), y)
            self._update_persistent_y(log_w_tilde, y_p, y_samples, idx, observed_mask)

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

    def _update_persistent_y(self, log_w_unnorm, y, y_samples, idx, observed_mask):
        """Sample new persistent y"""
        ys = concat_samples(y, y_samples)

        with torch.no_grad():
            sampled_idx = Categorical(logits=log_w_unnorm.transpose(1, 2)).sample()
            y_p = torch.gather(ys, dim=1, index=sampled_idx.unsqueeze(dim=1)).squeeze(dim=1)

            if self._persistent_y is None:
                assert y.shape[0] == self.batch_size
                assert idx == self.batch_size
                self._persistent_y = y.clone()
                #assert self._persistent_y.shape == y.shape

            # Update only for unobserved samples
            self._persistent_y[:idx, :] = self._persistent_y[:idx, :] * observed_mask + y_p.clone() * (1 - observed_mask)

        #assert y_p.shape == y.shape
            #for n, _ in enumerate(ys):
                #sampled_idx = Categorical(logits=log_w_unnorm[n, :, :].transpose(0, 1)).sample()
            #    self._persistent_y[idx[n].item()] = y_p[n, :]
            #   y_p = torch.gather(ys, dim=1, index=sampled_idx.unsqueeze(dim=1)).squeeze(dim=1)

