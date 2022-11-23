"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights


# TODO: Not inherit from PartFnestimator?
class NceBinaryCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        w_tilde = self._norm_w(y, y_samples)
        num_neg = y_samples.shape[0]
        return -torch.log(w_tilde[0] / (w_tilde[0] + num_neg)) + torch.log((num_neg / (w_tilde[1:] + num_neg)).sum())

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (binary version).

        Note that in the NCE binary criterion we use an extra parameter
        to estimate the partition function.
        """

        return self._unnorm_distr.log_part_fn

    def _norm_w(self, y, y_samples) -> Tensor:
        y = y.reshape((1,))
        ys = torch.cat((y, y_samples))
        return unnorm_weights(ys, self._norm_distr, self._noise_distr)
