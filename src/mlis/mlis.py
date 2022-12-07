"""Maximum Likelihood Importance Sampling (MLIS)

Approximate -log p_theta(y) ~= -log p_tilde_theta(y) + \log Ẑ_theta,
where
Ẑ_theta = 1/J sum_j p_tilde_theta(y_j)
"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights


class MlisCrit(PartFnEstimator):
    """Maximum Likelihood Importance Sampling (MLIS) criterion"""

    def __init__(self, unnorm_distr, noise_distr):
        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        """MLIS criterion"""
        return -torch.log(self._unnorm_distr(y)) + self.log_part_fn(None, y_samples)

    def part_fn(self, _y, y_samples) -> Tensor:
        # The number of samples is the number of negative samples
        w_tilde = self._unnorm_w(None, y_samples)
        return w_tilde.mean()

    def _unnorm_w(self, _y, y_samples) -> Tensor:
        """Compute unnormalised weights

        The result is a tensor of J elements, where
        w_tilde(y_j) = p_tilde(y_j) / p_n(y_j), for j = 1, ..., J
        """
        return unnorm_weights(
            y_samples, self._unnorm_distr.prob, self._noise_distr.prob
        )
