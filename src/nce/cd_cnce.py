"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, log_cond_unnorm_weights, extend_sample

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class CdCnceCrit(PartFnEstimator):
    def __init__(self, unnorm_distr: BaseModel, noise_distr: NoiseDistr, num_neg_samples: int,  mcmc_steps: int):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps

    def crit(self, y: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor):

        # Gradient of mean is same as mean of gradient (?)
        grads = - self._unnorm_distr.grad_log_prob(y)

        y_samples, w = [], []
        for t in range(self.mcmc_steps):
            # Get neg. samples

            if t > 0:
                # Sample y
                # TODO: How sample here when we have several neg. samples per y?
                sample_inds = torch.distributions.one_hot_categorical.OneHotCategorical(probs=w).sample((1,))
                y_0 = y_samples.reshape(y.shape[0], -1, y.shape[-1])[sample_inds, :]
            else:
                y_0 = y.clone()

            y_samples = self.sample_noise(self._num_neg * y.size(0), y_0)

            # Calculate and normalise weights
            w = self._unnorm_w(y, y_samples)

            # Calculate gradients of log prob
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y, w[:y.shape[0]])
            grads_log_prob_y_samples = self._unnorm_distr.grad_log_prob(y_samples, 1 - w[y.shape[0]:])

            # Sum over samples, mean over iter.
            # TODO: Saknar jag någon * J här?
            grads = [grad + (1/ self.mcmc_steps) * (grads_log_prob_y[i] + self._num_neg * grads_log_prob_y_samples[i])
                     for i, grad in enumerate(grads)]

        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:
        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples):
        """Normalised weights of y (NxD) and y_samples ((N*J)xD)"""

        y = torch.repeat_interleave(y, int(y_samples.size(0) / y.size(0)), dim=0)

        return log_cond_unnorm_weights(y, y_samples, self._unnorm_distr.log_prob, self._noise_distr.log_prob)