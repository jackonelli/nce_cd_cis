"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights, concat_samples, extend_sample

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class CdRankCrit(PartFnEstimator):
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
                sample_inds = torch.distributions.one_hot_categorical.OneHotCategorical(probs=w).sample((1,))
                y_0 = y_samples.reshape(y.shape[0], -1, y.shape[-1])[sample_inds, :]
            else:
                y_0 = y.clone()

            y_samples = self.sample_noise(self._num_neg * y.size(0), y_0)

            # Calculate and normalise weights
            w = self._norm_w(y, y_samples)

            # Calculate gradients of log prob
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(extend_sample(y, y_samples), w)

            # Sum over samples, mean over iter.
            grads = [grad + (self._num_neg / self.mcmc_steps) * grad_log_prob_y for grad, grad_log_prob_y in
                     zip(grads, grads_log_prob_y)]

        self._unnorm_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _unnorm_w(self, y, y_samples) -> Tensor:

        ys = extend_sample(y, y_samples)
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)

    def _norm_w(self, y, y_samples):
        """Normalised weights of y (NxD) and y_samples ((N*J)xD)"""

        # TODO: I would want to do this in a nicer way, but for now let's do it like this
        w_tilde = self._unnorm_w(y, y_samples)
        w_norm = [torch.cat((w_tilde[i], w_tilde[(self._num_neg * i):(self._num_neg * (i + 1))])) for i in
                  range(y.shape[0])]

        return torch.cat([w_n / w_n.sum() for w_n in w_norm])