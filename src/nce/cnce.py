"""Conditional Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_cond_unnorm_weights


class CondNceCrit(PartFnEstimator):

    def __init__(self, unnorm_distr, noise_distr, num_neg_samples: int):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def inner_crit(self, y:Tensor, y_samples):

        log_w_tilde = self._log_unnorm_w(y, y_samples)
        return torch.log(1 + torch.exp(- log_w_tilde)).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version). """
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:

        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples) -> Tensor:
        """Log weights of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights(y.reshape(y.size(0), 1, -1), y_samples, self._unnorm_distr.log_prob,
                                       self._noise_distr.log_prob)
