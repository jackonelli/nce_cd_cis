"""Maximum Likelihood Importance Sampling (MLIS) partition function"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import unnorm_weights


class MlisCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        return -torch.log(self._unnorm_distr(y)) + self.log_part_fn(None, y_samples)

    def part_fn(self, _y, y_samples) -> Tensor:
        # The number of samples is the number of negative samples
        w_tilde = self._unnorm_w(None, y_samples)
        return w_tilde.mean()

    def _unnorm_w(self, _y, y_samples) -> Tensor:
        return unnorm_weights(y_samples, self._unnorm_distr.prob, self._noise_distr.prob)
