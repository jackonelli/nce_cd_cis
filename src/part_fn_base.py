"""Partition function estimator interface"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class PartFnEstimator(ABC):
    def log_part_fn(self, y, y_samples) -> Tensor:
        return torch.log(self.log_part_fn(y, y_samples))

    @abstractmethod
    def part_fn(self, y, y_samples) -> Tensor:
        pass


def unnorm_weights(y, unnorm_distr, noise_distr):
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)
