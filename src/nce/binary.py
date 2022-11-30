"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights


class NceBinaryCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def crit(self, y: Tensor) -> Tensor:
        y_samples = self.sample_noise(self._num_neg * y.size(0), y)

        return self.inner_crit(y, y_samples)

    def inner_crit(self, y: Tensor, y_samples: Tensor):
        w_tilde = self._norm_w(y, y_samples)
        num_neg = y_samples.shape[0] / y.shape[0]

        return - torch.log(w_tilde[:y.shape[0]] / (w_tilde[:y.shape[0]] + num_neg)).mean() \
               - num_neg * torch.log((num_neg / (w_tilde[y.shape[0]:] + num_neg))).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (binary version).

        Note that in the NCE binary criterion we use an extra parameter
        to estimate the partition function.
        """

        return self._unnorm_distr.log_part_fn

    def _norm_w(self, y, y_samples) -> Tensor:

        if y.ndim == 1:
            y = y.reshape((1, -1))

        ys = torch.cat((y, y_samples))
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)


