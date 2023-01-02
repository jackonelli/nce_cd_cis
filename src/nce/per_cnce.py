"""Persistent Conditional Noise Contrastive Estimation (CNCE)

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.

The method works like CNCE (`CondNceCrit`) but it has a persistent sample yp
that is used to condition on when sampling noisy samples.
"""
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Categorical

from src.nce.cd_cnce import CdCnceCrit
from src.part_fn_utils import concat_samples


class PersistentCondNceCrit(CdCnceCrit):
    """Persistent cond. NCE crit"""

    def __init__(self, unnorm_distr, noise_distr, num_neg_samples: int):
        mcmc_steps = 1  #TODO: If we want to take several MCMC-steps, persistent y should  be updated at end of gradient calculation?
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, mcmc_steps)
        self._persistent_y = dict()

    def calculate_crit_grad(self, y: Tensor, idx: Optional[Tensor]) -> Tensor:
        assert (
            idx is not None
        ), "PersistentCondNceCrit requires an idx tensor that is not None"

        y_p = self.persistent_y(y, idx)
        y_samples = self.sample_noise(self._num_neg, y_p)
        # NB We recompute w_tilde in inner_crit to comply with the API.
        log_w_tilde = self._log_unnorm_w(y, y_samples)
        self._update_persistent_y(log_w_tilde, y_p, y_samples, idx)

        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        y_p = torch.repeat_interleave(y_p, self._num_neg, dim=0)

        assert torch.allclose(y_samples[0, :, :],
                              y_samples.reshape(-1, 1, y.shape[-1])[:self._num_neg, 0, :], rtol=1e-3)

        return self.calculate_inner_crit_grad(y_p, y_samples.reshape(-1, 1, y.shape[-1]), y)

    def persistent_y(self, actual_y: Tensor, idx: Tensor):
        """Get persistent y

        Access the last selected y or return the y sampled from p_d
        if no last selected y exists
        """
        per_y = torch.empty(actual_y.size())
        for n, per_n in enumerate(idx):
            per_n = per_n.item()
            per_y[n, :] = (
                self._persistent_y[per_n]
                if self._persistent_y.get(per_n) is not None
                else actual_y[n, :]
            )
        return per_y

    def _update_persistent_y(self, log_w_unnorm, y, y_samples, idx):
        """Sample new persistent y"""
        ys = concat_samples(y, y_samples)
        for n, _ in enumerate(ys):
            sampled_idx = Categorical(logits=log_w_unnorm[n, :]).sample()
            self._persistent_y[idx[n].item()] = ys[n, sampled_idx]
