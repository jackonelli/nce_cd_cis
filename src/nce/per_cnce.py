"""Persistent Noise Contrastive Estimation (NCE)

Inpsired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.
"""
from typing import Optional
import torch
from torch import Tensor
import numpy as np
from torch.distributions import Categorical

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_cond_unnorm_weights, concat_samples


class PersistentCondNceCrit(PartFnEstimator):
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
        return self.inner_crit(y, y_samples, idx)

    def inner_crit(self, y: Tensor, y_samples, _idx: Optional[Tensor]):

        log_w_tilde = self._log_unnorm_w(y, y_samples)
        return torch.log(1 + torch.exp(-log_w_tilde)).mean()

    def sample_noise(self, num_neg: int, y: Tensor):
        num_samples = y.size(0)
        return self._noise_distr.sample(
            torch.Size((num_samples, num_neg)), y.reshape(num_samples, 1, -1)
        )

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

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version)."""
        pass

    def _log_unnorm_w(self, y, y_samples) -> Tensor:
        """Log weights of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights(
            y.reshape(y.size(0), 1, -1),
            y_samples,
            self._unnorm_distr.log_prob,
            self._noise_distr.log_prob,
        )

    # def _unnorm_w(self, y, y_samples) -> Tensor:

    #     if y.ndim == 1:
    #         y = y.reshape((1, -1))

    #     assert np.remainder(y_samples.size(0), y.size(0)) == 0

    #     # Alternative to this: use broadcasting
    #     y = torch.repeat_interleave(y, int(y_samples.size(0) / y.size(0)), dim=0)

    #     return cond_unnorm_weights(
    #         y, y_samples, self._unnorm_distr.prob, self._noise_distr.prob
    #     )
