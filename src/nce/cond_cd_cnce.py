"""Conditional Noise Contrastive Estimation (NCE) with multiple MCMC steps for conditional model (RBM) """
from typing import Optional
import torch
from torch import Tensor
from src.nce.cd_cnce import CdCnceCrit
from src.part_fn_utils import cond_log_cond_unnorm_weights, cond_log_cond_unnorm_weights_ratio, concat_samples

from src.noise_distr.base import NoiseDistr
from src.models.rbm.conditional_rbm import CondRbm

from src.training.training_utils import add_to_npy_file


class CondCdCnceCrit(CdCnceCrit):
    def __init__(
        self,
        unnorm_distr: CondRbm,
        noise_distr: NoiseDistr,
        num_neg_samples: int,
        mcmc_steps: int,
        save_acc_prob=False,
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, mcmc_steps, save_acc_prob)

    def calculate_crit_grad(self, y: tuple, _idx: Optional[Tensor]):
        # We will have N*J pairs

        y, x = y
        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        x = torch.repeat_interleave(x, self._num_neg, dim=0)
        y_samples = self.sample_noise(1, y)

        return self.calculate_inner_crit_grad((y, x), y_samples)

    def calculate_inner_crit_grad(self, y: tuple, y_samples: Tensor, y_base=None, log_w_thrs=100):

        y, x = y
        if y_base is None:
            # Gradient of mean is same as mean of gradient
            grads_log_prob_y = self._unnorm_distr.grad_log_prob((y, x))
        else:
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y_base)

        grads = [-grad_log_prob_y for grad_log_prob_y in grads_log_prob_y]

        y_0 = y.clone()
        for t in range(self.mcmc_steps):

            # Concatenate samples
            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y.shape[0], 2, y.shape[1])

            # Calculate and normalise weight ratios
            log_w_y = self._log_unnorm_w_ratio((y_0, x), y_samples).detach()
            #log_w_y[log_w_y < log_w_thrs] = log_w_thrs

            w_y = 1 / (1 + torch.exp(-log_w_y))
            w = torch.cat((w_y, 1 - w_y), dim=1)

            if self.save_acc_prob:
                # Ref. MH acceptance prob.
                acc_prob_mh = torch.exp(- log_w_y)
                acc_prob_mh[acc_prob_mh >= 1.0] = 1.0
                add_to_npy_file("res/" + "cd_cnce_num_neg_" + str(self._num_neg) + "_cd_cnce_acc_prob.npy",
                                (1 - w_y).numpy())
                add_to_npy_file("res/" + "cd_cnce_num_neg_" + str(self._num_neg) + "_cd_mh_acc_prob.npy",
                                acc_prob_mh.numpy())

            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob((ys, x.unsqueeze(dim=1)), w)

            # Sum over samples (2), mean over iter.
            grads = [
                grad + (2 / self.mcmc_steps) * grad_log_prob
                for grad, grad_log_prob in zip(grads, grads_log_prob)
            ]

            if (t + 1) < self.mcmc_steps:
                # Sample y
                y_0, y_samples = self._resample(ys, w_y)
                assert y_0.shape == y.shape

        self._unnorm_distr.set_gradients(grads)

    def log_part_fn(self, y: tuple, _idx: Optional[Tensor]) -> Tensor:
        y, x = y
        y = torch.repeat_interleave(y, self._num_neg, dim=0)
        x = torch.repeat_interleave(x, self._num_neg, dim=0)
        y_samples = self.sample_noise(1, y)

        return self.inner_log_part_fn((y, x), y_samples)

    def inner_log_part_fn(self, y: tuple, y_samples) -> Tensor:

        y, x = y
        y_0 = y.clone()

        log_z = 0
        for t in range(self.mcmc_steps):

            ys = concat_samples(y_0, y_samples)
            assert ys.shape == (y.shape[0], 2, y.shape[1])

            log_w_y = self._log_unnorm_w_ratio((y_0, x), y_samples)
            w_y = 1 / (1 + torch.exp(-log_w_y))
            w = torch.cat((w_y, 1 - w_y), dim=1)

            log_z += ((w * self._unnorm_distr.log_prob((ys, x.unsqueeze(dim=1)))).sum(dim=1)).mean()  # TODO: Normalised weights, right?

            if (t + 1) < self.mcmc_steps:
                # Sample y
                y_0, y_samples = self._resample(ys, w_y)
                assert y_0.shape == y.shape

        return log_z / torch.tensor(self.mcmc_steps)

    def _log_unnorm_w(self, y, y_samples):
        # "Log weights of y (NxD) and y_samples (NxJxD)"
        # Note: dimension of the final weight vector is NxJx2

        y, x = y

        w_tilde_y = cond_log_cond_unnorm_weights(y.reshape(y.size(0), 1, -1), x.reshape(y.size(0), 1, -1), y_samples,
                                                 self._unnorm_distr.log_prob, self._noise_distr.log_prob)

        w_tilde_yp = cond_log_cond_unnorm_weights(y_samples, x.reshape(y.size(0), 1, -1), y.reshape(y.size(0), 1, -1),
                                                  self._unnorm_distr.log_prob, self._noise_distr.log_prob)

        return torch.stack((w_tilde_y, w_tilde_yp), dim=-1)

    def _log_unnorm_w_ratio(self, y, y_samples):
        """Log weight ratio of y (NxD) and y_samples (NxJxD)"""

        y, x = y
        return cond_log_cond_unnorm_weights_ratio(
                y.reshape(y.size(0), 1, -1),
                x.reshape(y.size(0), 1, -1),
                y_samples,
                self._unnorm_distr.log_prob,
                self._noise_distr.log_prob)






