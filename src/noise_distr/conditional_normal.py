"""Normal (Gaussian) noise distributions"""
import torch
from torch import Tensor
import numpy as np
from src.noise_distr.base import NoiseDistr


class ConditionalNormal(NoiseDistr):
    """Normal distr. with adaptive mean"""

    def __init__(self, sigma_sq: float):
        self.sigma_sq = sigma_sq
        self._inner_distr = torch.distributions.Normal(0, np.sqrt(sigma_sq))

    def sample(self, size: torch.Size, x: Tensor):
        return self._inner_distr.rsample(size) + x

    def log_prob(self, samples, x: Tensor):
        return self._inner_distr.log_prob(samples - x)


class ConditionalMultivariateNormal(NoiseDistr):
    """Multivariate normal distr. with adaptive mean"""

    def __init__(self, cov: Tensor):
        self.cov = cov
        self.dim = cov.shape[0]
        self._inner_distr = torch.distributions.MultivariateNormal(torch.zeros(self.dim), cov)

    def sample(self, size: torch.Size, x: Tensor):
        assert x.size(-1) == self.dim

        x = torch.repeat_interleave(x, int(size[0] / x.size(0)), dim=0)
        assert x.size(0) == size[0]
        eps = self._inner_distr.rsample(size) 

        return x + eps

    def log_prob(self, samples, x: Tensor):
        assert samples.size(-1) == self.dim and x.size(-1) == self.dim
        return self._inner_distr.log_prob(samples - x)

