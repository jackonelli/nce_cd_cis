"""Criteria for training of RBMs"""
from typing import Optional
from torch import Tensor
from src.part_fn_base import PartFnEstimator

from src.noise_distr.base import NoiseDistr
from src.models.rbm.rbm import Rbm


class RbmCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: Rbm,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps

    def crit(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        _, y_sample = self._unnorm_distr.sample(y, k=self.mcmc_steps)

        return self.inner_crit(y, y_sample)

    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        return (self._unnorm_distr.log_prob(y_samples) - self._unnorm_distr.log_prob(y)).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass