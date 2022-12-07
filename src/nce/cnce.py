"""Conditional Noise Contrastive Estimation (NCE) partition functions"""
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_cond_unnorm_weights


class CondNceCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples: int):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def crit(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise((y.size(0), self._num_neg), y)

        return self.inner_crit(y, y_samples, _idx)

    def inner_crit(self, y: Tensor, y_samples, _idx: Optional[Tensor]):

        log_w_tilde = self._log_unnorm_w(y, y_samples)
        return torch.log(1 + torch.exp(-log_w_tilde)).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version)."""
        pass

    def sample_noise(self, num_samples: tuple, y: Tensor):
        return self._noise_distr.sample(
            torch.Size(num_samples), y.reshape(y.size(0), 1, -1)
        )

    def _unnorm_w(self, y, y_samples) -> Tensor:

        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples) -> Tensor:
        """Log weights of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights(
            y.reshape(y.size(0), 1, -1),
            y_samples,
            self._unnorm_distr.log_prob,
            self._noise_distr.log_prob,
        )
