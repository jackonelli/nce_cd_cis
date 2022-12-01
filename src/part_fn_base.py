"""Partition function estimator interface"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class PartFnEstimator(ABC):
    def __init__(self, unnorm_distr: BaseModel, noise_distr: NoiseDistr, num_neg_samples: int):

        self._unnorm_distr = unnorm_distr
        self._noise_distr = noise_distr
        self._num_neg = num_neg_samples

    def log_part_fn(self, y, y_samples) -> Tensor:
        return torch.log(self.log_part_fn(y, y_samples))

    @abstractmethod
    def crit(self, y: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        self.crit(y).backward()

    @abstractmethod
    def part_fn(self, y, y_samples) -> Tensor:
        pass

    def sample_noise(self, num_samples: int, y: Tensor):
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
    """Combine one vector y with set of many vectors y_1:J into y_0:J"""

    return torch.cat((y, y_sample))


def norm_weights(unnorm_weights: Tensor) -> Tensor:
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j) for all y_j"""
    return unnorm_weights / unnorm_weights.sum()


def concat_samples(y: Tensor, y_samples: Tensor) -> Tensor:
    """Concatenate y (NxD), y_samples (N*JxD) and reshape as Nx(J+1)xD"""

    y_reshaped = y.reshape(y.shape[0], 1, -1)
    return torch.cat((y_reshaped, y_samples.reshape(y_reshaped.shape[0], -1, y_reshaped.shape[-1])), dim=1)
