"""Conditional Noise Contrastive Estimation (NCE) with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_cond_unnorm_weights, concat_samples

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class CdCnceCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: BaseModel,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps

    def inner_crit(
        self, y: Tensor, y_samples: Tensor, _idx: Optional[Tensor]
    ) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        # We will have N*J pairs
        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        y_samples = self.sample_noise((y.size(0), 1), y)

        return self.calculate_inner_crit_grad(y, y_samples, _idx)

    def calculate_inner_crit_grad(
        self, y: Tensor, y_samples: Tensor, _idx: Optional[Tensor]
    ):

        # Gradient of mean is same as mean of gradient
        grads_log_prob_y = self._unnorm_distr.grad_log_prob(y)
        grads = [-grad_log_prob_y for grad_log_prob_y in grads_log_prob_y]

        y_0 = y.clone()
        log_w_threshold = 10
        w_y = torch.zeros((y.shape[0], 1), dtype=y.dtype)
        for t in range(self.mcmc_steps):

            # Get neg. samples
            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y_0.size(0), 2, y_0.size(1))

            # Calculate and normalise weights
            log_w_y = self._log_unnorm_w(y_0, y_samples).detach()
            w_y[log_w_y <= log_w_threshold] = 1 / (
                1 + torch.exp(-log_w_y[log_w_y <= log_w_threshold])
            )

            # For computational stability
            w_y[log_w_y > log_w_threshold] = 1.0
            w = torch.cat((w_y, 1 - w_y), dim=1)

            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob(ys, w)

            # Sum over samples (2), mean over iter.
            grads = [
                grad + (2 / self.mcmc_steps) * grad_log_prob
                for grad, grad_log_prob in zip(grads, grads_log_prob)
            ]

            if (t + 1) < self.mcmc_steps:
                # Sample y
                sample_inds = torch.distributions.bernoulli.Bernoulli(
                    probs=1 - w_y
                ).sample()
                y_0 = ys[torch.cat((sample_inds, 1 - sample_inds), dim=-1).bool(), :]

                assert y_0.shape == y.shape

                # Sample neg. samples
                y_samples = self.sample_noise((y_0.size(0), 1), y_0)

        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def sample_noise(self, num_samples: tuple, y: Tensor):
        return self._noise_distr.sample(
            torch.Size(num_samples), y.reshape(y.size(0), 1, -1)
        )

    def _unnorm_w(self, y, y_samples) -> Tensor:
        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples):
        """Log weights of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights(
            y.reshape(y.size(0), 1, -1),
            y_samples,
            self._unnorm_distr.log_prob,
            self._noise_distr.log_prob,
        )
