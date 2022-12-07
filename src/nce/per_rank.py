"""Persistent Noise Contrastive Estimation (NCE) for the ranking criterion

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.

The method works like CNCE (`CondNceCrit`) but it has a persistent sample yp
that is used to condition on when sampling noisy samples.
"""

from typing import Optional
import torch
from torch import Tensor
from src.nce.rank import NceRankCrit
from src.part_fn_utils import concat_samples
from torch.distributions import Categorical


class PersistentNceRankCrit(NceRankCrit):
    """Persistent rank. NCE crit"""

    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)
        self._persistent_y = dict()

    def crit(self, y: Tensor, idx: Optional[Tensor]) -> Tensor:
        assert (
            idx is not None
        ), "PersistentNceRankCrit requires an idx tensor that is not None"
        yp = self.persistent_y(y, idx)
        y_samples = self.sample_noise((y.size(0), self._num_neg), yp)

        return self.inner_crit(yp, y_samples, None)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (ranking version).

        Note that in the NCE ranking criterion we use a scaled version of Ẑ,
        L_NCE = -log(w_tilde(y_0)) + log ( (J+1) Ẑ),
        though it has no practical effect on the gradients.
        """
        # TODO: idx parameter
        return super().part_fn(y, y_samples)

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
