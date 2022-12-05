"""Noise Contrastive Estimation (NCE) partition functions"""
from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import unnorm_weights, concat_samples


class NceBinaryCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def crit(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise((y.size(0), self._num_neg), y)

        return self.inner_crit(y, y_samples)

    def inner_crit(self, y: Tensor, y_samples: Tensor, _idx: Optional[Tensor]):
        w = self._norm_w(y, y_samples)

        return - torch.log(w[:, 0] / (w[:, 0] + self._num_neg)).mean() \
               - self._num_neg * torch.log((self._num_neg / (w[:, 1:] + self._num_neg))).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (binary version).

        Note that in the NCE binary criterion we use an extra parameter
        to estimate the partition function.
        """

        return self._unnorm_distr.log_part_fn

    def _norm_w(self, y, y_samples) -> Tensor:
        """Calculate weights: return as (Nx(J+1)) tensor"""

        ys = concat_samples(y, y_samples)
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)
