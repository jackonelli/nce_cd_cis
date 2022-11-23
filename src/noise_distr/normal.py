"""Normal (Gaussian) noise distributions"""
import torch
import numpy as np
from src.noise_distr.base import NoiseDistr


class Normal(NoiseDistr):
    def __init__(self, mu: float, sigma_sq: float):
        self.mu = mu
        self.sigma_sq = sigma_sq
        self._inner_distr = torch.distributions.Normal(mu, np.sqrt(sigma_sq))

    def sample(self, size: torch.Size, x: torch.Tensor):
        return self._inner_distr.rsample(size)

    def log_prob(self, samples, x):
        return self._inner_distr.log_prob(samples)
