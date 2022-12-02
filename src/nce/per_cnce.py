"""Persistent Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
import numpy as np

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import cond_unnorm_weights


class PersistentCondNceCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)
        self._persistent_y = None

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        y_p = self.persistent_y(y)
        y_samples = self.sample_noise(self._num_neg * y_p.size(0), y_p)
        w_tilde = self._unnorm_w(y, y_samples)
        self._update_persistent_y(w_tilde, y, y_samples)
        return torch.log(1 + (1 / w_tilde)).mean()

    def persistent_y(self, actual_y: Tensor):
        """Get persistent y

        Access the last selected y or return the y sampled from p_d
        if no last selected y exists
        """
        per_y = self._persistent_y if self._persistent_y is not None else actual_y
        return per_y

    def _update_persistent_y(self, w_unnorm, y, y_samples):
        """Sample new persistent y"""
        w_norm = w_unnorm / w_unnorm.sum(1)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version)."""
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:

        if y.ndim == 1:
            y = y.reshape((1, -1))

        assert np.remainder(y_samples.size(0), y.size(0)) == 0

        # Alternative to this: use broadcasting
        y = torch.repeat_interleave(y, int(y_samples.size(0) / y.size(0)), dim=0)

        return cond_unnorm_weights(
            y, y_samples, self._unnorm_distr.prob, self._noise_distr.prob
        )
