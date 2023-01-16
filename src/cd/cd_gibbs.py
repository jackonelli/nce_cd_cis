"""Contrastive divergence (CD) with Gibbs sampling (for RBMs) """
# TODO: This is according to Wikipedia: https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine. It has only
#  implemented as a temporary reference, if we want to use it in experiments later on,
#  we should probably check its validity
from typing import Optional

import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import outer_product
from src.noise_distr.base import NoiseDistr
from src.models.rbm.rbm import Rbm


class CdGibbsCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: Rbm,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps

    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):

        h, y_sample, h_sample = self.gibbs_sample(y)
        return self.calculate_inner_crit_grad((y, h), (y_sample, h_sample))

    def calculate_inner_crit_grad(self, y, y_samples):

        y, h = y
        y_sample, h_sample = y_samples

        grad_weights, grad_vis_bias, grad_hidden_bias = 0, 0, 0

        y_0 = y.clone()
        for t in range(self.mcmc_steps):

            grad_weights -= (outer_product(y_0, h) - outer_product(y_sample, h_sample)).mean(dim=0)
            grad_vis_bias -= (y_0 - y_sample).mean(dim=0).reshape(-1, 1)
            grad_hidden_bias -= (h - h_sample).mean(dim=0).reshape(-1, 1)

            if (t + 1) < self.mcmc_steps:
                y_0 = y_sample.clone()
                h, y_sample, h_sample = self.gibbs_sample(y_0)

        grads = [grad_weights / self.mcmc_steps, grad_vis_bias / self.mcmc_steps, grad_hidden_bias / self.mcmc_steps]

        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def gibbs_sample(self, y: Tensor):

        # TODO: If we keep this, we should have a separate noise distribution instead
        _, h = self._unnorm_distr.sample_hidden(y)
        _, y_sample = self._unnorm_distr.sample_visible(h)
        _, h_sample = self._unnorm_distr.sample_hidden(y_sample)

        return h.detach().clone(), y_sample.detach().clone(), h_sample.detach().clone()

