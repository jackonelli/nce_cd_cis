"""Contrastive Divergence with Metropolis-Hastings kernel"""
import os
from typing import Optional
from pathlib import Path
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.utils.part_fn_utils import log_cond_unnorm_weights, log_cond_unnorm_weights_ratio, concat_samples

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel

from src.training.training_utils import add_to_npy_file


class CdMHCnceCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: BaseModel,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
        save_acc_prob=False,
	save_dir=None, # Path
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps
        self.save_acc_prob = save_acc_prob
        self.save_dir = save_dir if save_dir is not None else "res/"
        self.name = "cd_mh"

    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        # We will have N*J pairs
        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        y_samples = self.sample_noise(1, y) #self.sample_noise((y.size(0), 1), y)

        return self.calculate_inner_crit_grad(y, y_samples)

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor, y_base=None):

        # Gradient of mean is same as mean of gradient
        if y_base is None:
            # Gradient of mean is same as mean of gradient
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y)
            grads = [-grad_log_prob_y / y.shape[0] for grad_log_prob_y in grads_log_prob_y]
        else:
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y_base)
            grads = [-grad_log_prob_y / y_base.shape[0] for grad_log_prob_y in grads_log_prob_y]

        y_0 = y.clone()
        for t in range(self.mcmc_steps):

            # Get neg. samples
            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y_0.size(0), 2, y_0.size(1))

            # Calculate weight ratios (acceptance prob.)
            log_w_y = self._log_unnorm_w_ratio(y_0, y_samples).detach()
            w_y = torch.exp(- log_w_y)
            w_y[w_y >= 1.0] = 1.0
            w = torch.cat((1 - w_y, w_y), dim=1)

            if self.save_acc_prob:
                # Ref. CNCE acc. prob.
                acc_prob_cnce = 1 - 1 / (1 + torch.exp(-log_w_y))
                add_to_npy_file(os.path.join(self.save_dir, self.name + "_num_neg_" + str(self._num_neg) + "_cd_mh_acc_prob.npy"), w_y.numpy())
                add_to_npy_file(os.path.join(self.save_dir, self.name + "_num_neg_" + str(self._num_neg) + "_cd_cnce_acc_prob.npy"), acc_prob_cnce.numpy())


            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob(ys, w)

            # Sum over samples (2), mean over iter.
            grads = [
                grad + (1 / (y.shape[0] * self.mcmc_steps)) * grad_log_prob
                for grad, grad_log_prob in zip(grads, grads_log_prob)
            ]

            if (t + 1) < self.mcmc_steps:
                # Sample y
                sample_inds = torch.distributions.bernoulli.Bernoulli(
                    probs=w_y
                ).sample()
                y_0 = ys[torch.cat((1 - sample_inds, sample_inds), dim=-1).bool(), :]

                assert y_0.shape == y.shape

                # Sample neg. samples
                y_samples = self.sample_noise(1, y)  # self.sample_noise((y_0.size(0), 1), y_0)

        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    #def sample_noise(self, num_samples: tuple, y: Tensor):
    #    return self._noise_distr.sample(
     #       torch.Size(num_samples), y.reshape(y.size(0), 1, -1)
     #   )

    def _unnorm_w(self, y, y_samples) -> Tensor:
        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples):

        w_tilde_y = log_cond_unnorm_weights(y.reshape(y.size(0), 1, -1), y_samples, self._unnorm_distr.log_prob,
                                            self._noise_distr.log_prob)
        w_tilde_yp = log_cond_unnorm_weights(y_samples, y.reshape(y.size(0), 1, -1), self._unnorm_distr.log_prob,
                                             self._noise_distr.log_prob)

        return torch.stack((w_tilde_y, w_tilde_yp), dim=-1)

    def _log_unnorm_w_ratio(self, y, y_samples):
        """Log weights of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights_ratio(
            y.reshape(y.size(0), 1, -1),
            y_samples,
            self._unnorm_distr.log_prob,
            self._noise_distr.log_prob,
        )
