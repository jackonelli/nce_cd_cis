"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
import numpy as np

from src.part_fn_base import PartFnEstimator, cond_unnorm_weights, log_cond_unnorm_weights


class CondNceCrit(PartFnEstimator):

    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        log_w_tilde = self._log_unnorm_w(y, y_samples)

        return torch.log(1 + torch.exp(- log_w_tilde)).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version).
        """
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:
        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples) -> Tensor:

        if y.ndim == 1:
            y = y.reshape((1, -1))

        assert np.remainder(y_samples.size(0), y.size(0)) == 0

        # Alternative to this: use broadcasting
        y = torch.repeat_interleave(y, int(y_samples.size(0) / y.size(0)), dim=0)

        return log_cond_unnorm_weights(y, y_samples, self._unnorm_distr.log_prob, self._noise_distr.log_prob)
