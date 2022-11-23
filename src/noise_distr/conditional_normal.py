"""Normal (Gaussian) noise distributions"""
import torch
import numpy as np
from src.noise_distr.base import NoiseDistr


class ConditionalNormal(NoiseDistr):
    """Normal distr. around observed x"""

    def __init__(self, sigma_sq: float):
        self.sigma_sq = sigma_sq
        self._inner_distr = torch.distributions.Normal(0, np.sqrt(sigma_sq))

    def sample(self, size: torch.Size, x: torch.Tensor):
        return self._inner_distr.rsample(size) + x

    def log_prob(self, samples, x):
        return self._inner_distr.log_prob(samples - x)



