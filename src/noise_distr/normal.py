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

    def log_prob(self, samples, x=0):
        return self._inner_distr.log_prob(samples)


class MultivariateNormal(NoiseDistr):
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self.mu = mu
        self.cov = cov
        self.dim = mu.size(0)
        self._inner_distr = torch.distributions.MultivariateNormal(mu, cov)

    def sample(self, size: torch.Size, x=0):
        return self._inner_distr.rsample(size)

    def log_prob(self, samples, x=0):
        assert samples.size(-1) == self.dim
        return self._inner_distr.log_prob(samples)




