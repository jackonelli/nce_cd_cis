"""Noise Contrastive Estimation (NCE) ranking partition functions with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import unnorm_weights, concat_samples

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class CdRankCrit(PartFnEstimator):
    def __init__(self, unnorm_distr: BaseModel, noise_distr: NoiseDistr, num_neg_samples: int,  mcmc_steps: int):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps

    def inner_crit(self, y: Tensor, y_samples: Tensor, _idx: Optional[Tensor]) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        y_samples = self.sample_noise((y.size(0), self._num_neg), y)

        return self.calculate_inner_crit_grad(y, y_samples, _idx)

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor, _idx: Optional[Tensor]):

        # Gradient of mean is same as mean of gradient (?)
        grads_log_prob_y = self._unnorm_distr.grad_log_prob(y)
        grads = [-grad_log_prob_y for grad_log_prob_y in grads_log_prob_y]

        y_0 = y.clone()
        for t in range(self.mcmc_steps):

            # Get neg. samples
            ys = concat_samples(y_0, y_samples)

            # Calculate and normalise weights
            w = self._norm_w(y_0, y_samples).detach()

            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob(ys, w)

            # Sum over samples, mean over iter.
            grads = [grad + ((self._num_neg + 1) / self.mcmc_steps) * grad_log_prob for grad, grad_log_prob in
                     zip(grads, grads_log_prob)]

            if (t + 1) < self.mcmc_steps:
                # Sample y for next step
                sample_inds = torch.distributions.one_hot_categorical.OneHotCategorical(probs=w).sample((1,))
                y_0 = ys[sample_inds, :]

                # Sample neg. samples
                y_samples = self.sample_noise((y_0.size(0), self._num_neg), y_0)

        # Assign calculated gradients to model parameters
        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:

        ys = concat_samples(y, y_samples)
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)

    def _norm_w(self, y, y_samples):
        """Normalised weights of y (NxD) and y_samples (NxJxD)"""

        w_tilde = self._unnorm_w(y, y_samples)
        return w_tilde / w_tilde.sum(dim=1, keepdim=True)