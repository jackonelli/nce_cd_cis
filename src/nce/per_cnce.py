"""Persistent Conditional Noise Contrastive Estimation (CNCE)

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.

The method works like CNCE (`CondNceCrit`) but it has a persistent sample yp
that is used to condition on when sampling noisy samples.
"""
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Categorical

from src.nce.cnce import CondNceCrit
from src.part_fn_utils import concat_samples


class PersistentCondNceCrit(CondNceCrit):
    """Persistent cond. NCE crit"""

    def __init__(self, unnorm_distr, noise_distr, num_neg_samples: int):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)
        self._persistent_y = dict()

    def crit(self, y: Tensor, idx: Optional[Tensor]) -> Tensor:
        assert (
            idx is not None
        ), "PersistentCondNceCrit requires an idx tensor that is not None"
        y_p = self.persistent_y(y, idx)
        y_samples = self.sample_noise(self._num_neg, y_p)
        # NB We recompute w_tilde in inner_crit to comply with the API.
        log_w_tilde = self._log_unnorm_w(y, y_samples)
        self._update_persistent_y(log_w_tilde, y_p, y_samples, idx)
        return self.inner_crit(y, y_samples)

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
        w_unnorm = torch.exp(log_w_unnorm)
        ys = concat_samples(y, y_samples)
        for n, _ in enumerate(ys):
            sampled_idx = Categorical(logits=w_unnorm[n, :]).sample()
            self._persistent_y[idx[n].item()] = ys[n, sampled_idx]
