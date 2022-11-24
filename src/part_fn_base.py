"""Partition function estimator interface"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class PartFnEstimator(ABC):
    def __init__(self, unnorm_distr, noise_distr):

        self._unnorm_distr = unnorm_distr
        self._noise_distr = noise_distr

    def log_part_fn(self, y, y_samples) -> Tensor:
        return torch.log(self.log_part_fn(y, y_samples))

    @abstractmethod
    def part_fn(self, y, y_samples) -> Tensor:
        pass

    def sample_noise(self, num_samples, y):
        return self._noise_distr.sample(torch.Size((num_samples,)), y)

    def get_model(self):
        return self._unnorm_distr


def unnorm_weights(y, unnorm_distr, noise_distr):
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)


def cond_unnorm_weights(y, yp, unnorm_distr, noise_distr) -> Tensor:
    return unnorm_distr(y) * noise_distr(y, yp) / (unnorm_distr(yp) * noise_distr(yp, y))
