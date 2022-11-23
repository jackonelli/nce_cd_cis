"""Noise distribution interface"""
from abc import ABC, abstractmethod
import torch
from torch import Tensor


class NoiseDistr(ABC):
    @abstractmethod
    def sample(self, size: torch.Size, x: torch.Tensor) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self, samples, x) -> Tensor:
        """Compute the log. probability of a sample conditional on obs. x.

        Most APIs expose a log_prob method for distributions.
        We conform to this and then provide a blanket impl for the method below.
        """
        pass

    def prob(self, samples, x=0):
        """Probability of a sample y  conditional on obs. x"""
        return torch.exp(self.log_prob(samples - x))


def unnorm_weights(y, unnorm_distr, noise_distr):
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)
