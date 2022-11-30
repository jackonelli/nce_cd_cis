"""Partition function estimator interface"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class PartFnEstimator(ABC):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):

        self._unnorm_distr = unnorm_distr
        self._noise_distr = noise_distr
        self._num_neg = num_neg_samples

    def log_part_fn(self, y, y_samples) -> Tensor:
        return torch.log(self.log_part_fn(y, y_samples))

    @abstractmethod
    def part_fn(self, y, y_samples) -> Tensor:
        pass

    def sample_noise(self, num_samples, y):
        return self._noise_distr.sample(torch.Size((num_samples,)), y)

    def get_model(self):
        return self._unnorm_distr


def unnorm_weights(y: Tensor, unnorm_distr, noise_distr) -> Tensor:
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)


def cond_unnorm_weights(y, yp, unnorm_distr, noise_distr) -> Tensor:
    return (
        unnorm_distr(y) * noise_distr(yp, y) / (unnorm_distr(yp) * noise_distr(y, yp))
    )


def log_cond_unnorm_weights(y, yp, log_unnorm_distr, log_noise_distr) -> Tensor:
    return (
        log_unnorm_distr(y)
        + log_noise_distr(yp, y)
        - log_unnorm_distr(yp)
        - log_noise_distr(y, yp)
    )


def extend_sample(y: Tensor, y_sample: Tensor) -> Tensor:
    """Combine actual samples with noisy ones

    Actual samples: y_1:N, tensor of shape (N, D)
    Noisy samples: y_1:J, for every y_n, n = 1, ..., N, tensor of shape (J, D)

    vector y with set of many vectors y_1:J into y_0:J"""

    return torch.cat((y, y_sample))


def norm_weights(unnorm_weights: Tensor) -> Tensor:
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j) for all y_j"""
    return unnorm_weights / unnorm_weights.sum()


# This is a bit buggy? It only computes it for y.
def old_norm_weights(y, y_samples, true_distr, noise_distr):
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j)"""
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (
        y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum()
    )
