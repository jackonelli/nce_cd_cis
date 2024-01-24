"""Conditional Noise Contrastive Estimation (NCE) with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_cond_unnorm_weights, log_cond_unnorm_weights_ratio, concat_samples

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel

from src.training.training_utils import add_to_npy_file


class CdCnceCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: BaseModel,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
        save_acc_prob=False,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps
        self.save_acc_prob = save_acc_prob
        self.name = "cd_cnce"

    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        # We will have N*J pairs
        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        y_samples = self.sample_noise(1, y)

        return self.calculate_inner_crit_grad(y, y_samples)

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor, y_base=None):

        if y_base is None:
            # Gradient of mean is same as mean of gradient
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y)
            grads = [-2 * grad_log_prob_y / y.shape[0]  for grad_log_prob_y in grads_log_prob_y]
        else:
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y_base)
            grads = [-2 * grad_log_prob_y / y_base.shape[0]  for grad_log_prob_y in grads_log_prob_y]

        for i, grad in enumerate(grads):
            print("grad y " + str(i))
            print(grad)

        y_0 = y.clone()
        for t in range(self.mcmc_steps):

            # Concatenate samples
            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y.shape[0], 2, y.shape[1])

            print("ys")
            print(ys)

            # Calculate and normalise weight ratios
            log_w_y = self._log_unnorm_w_ratio(y_0, y_samples).detach()
            w_y = 1 / (1 + torch.exp(-log_w_y))

            print("weights")
            print(w_y)

            w = torch.cat((w_y, 1 - w_y), dim=1)

            if self.save_acc_prob:
                # Ref. MH acceptance prob.
                acc_prob_mh = torch.exp(- log_w_y)
                acc_prob_mh[acc_prob_mh >= 1.0] = 1.0
                add_to_npy_file("res/" + self.name + "_num_neg_" + str(self._num_neg) + "_cd_cnce_acc_prob.npy",
                                (1 - w_y).numpy())
                add_to_npy_file("res/" + self.name + "_num_neg_" + str(self._num_neg) + "_cd_mh_acc_prob.npy",
                                acc_prob_mh.numpy())


            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob(ys, w)

            for i, grad in enumerate(grads_log_prob):
                print("grad ys " + str(i))
                print(grad)

            # Mean over samples, mean over iter.
            print(self.mcmc_steps)
            grads = [
                grad + (1 / (y_0.shape[0] * self.mcmc_steps)) * grad_log_prob
                for grad, grad_log_prob in zip(grads, grads_log_prob)
            ]

            if (t + 1) < self.mcmc_steps:
                # Sample y
                y_0, y_samples = self._resample(ys, w_y)
                assert y_0.shape == y.shape

        self._unnorm_distr.set_gradients(grads)

    def _resample(self, ys, w_y):
        # Sample y
        sample_inds = torch.distributions.bernoulli.Bernoulli(
            probs=1 - w_y
        ).sample()
        y = ys[torch.cat((1 - sample_inds, sample_inds), dim=-1).bool(), :]

        # Sample neg. samples
        y_samples = self.sample_noise(1, y)

        return y, y_samples

    def log_part_fn(self, y, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise(self._num_neg, y.reshape(y.size(0), 1, -1))

        return self.inner_log_part_fn(y, y_samples)

    def inner_log_part_fn(self, y, y_samples) -> Tensor:

        y_0 = y.clone()
        log_z = 0
        for t in range(self.mcmc_steps):

            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y.shape[0], 2, y.shape[1])

            log_w_y = self._log_unnorm_w_ratio(y_0, y_samples)
            w_y = 1 / (1 + torch.exp(-log_w_y))
            w = torch.cat((w_y, 1 - w_y), dim=1)

            log_z += ((w * self._unnorm_distr.log_prob(ys)).sum(dim=1)).mean()

            if (t + 1) < self.mcmc_steps:
                # Sample y
                y_0, y_samples = self._resample(ys, w_y)
                assert y_0.shape == y.shape

        return log_z / torch.tensor(self.mcmc_steps)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:
        return torch.exp(self._log_unnorm_w(y, y_samples))

    def _log_unnorm_w(self, y, y_samples):
        # "Log weights of y (NxD) and y_samples (NxJxD)"
        # Note: dimension of the final weight vector is NxJx2

        w_tilde_y = log_cond_unnorm_weights(y.reshape(y.size(0), 1, -1), y_samples, self._unnorm_distr.log_prob,
                                            self._noise_distr.log_prob)
        w_tilde_yp = log_cond_unnorm_weights(y_samples, y.reshape(y.size(0), 1, -1), self._unnorm_distr.log_prob,
                                             self._noise_distr.log_prob)

        return torch.stack((w_tilde_y, w_tilde_yp), dim=-1)

    def _log_unnorm_w_ratio(self, y, y_samples):
        """Log weight ratio of y (NxD) and y_samples (NxJxD)"""

        return log_cond_unnorm_weights_ratio(
            y.reshape(y.size(0), 1, -1),
            y_samples,
            self._unnorm_distr.log_prob,
            self._noise_distr.log_prob,
        )
