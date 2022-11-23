"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, cond_unnorm_weights


class CondNceCrit(PartFnEstimator):

    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        w_tilde = self._unnorm_w(y, y_samples)
        return torch.log(1 + (1 / w_tilde)).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (conditional version).
        """
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:
        y = y.reshape((1,))
        return cond_unnorm_weights(y, y_samples, self._unnorm_distr, self._noise_distr)
