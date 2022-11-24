"""Normal (Gaussian) noise distributions"""
import torch
import numpy as np
from src.noise_distr.base import NoiseDistr


class ConditionalNormal(NoiseDistr):
    """Normal distr. with adaptive mean"""

    def __init__(self, sigma_sq: float):
        self.sigma_sq = sigma_sq
        self._inner_distr = torch.distributions.Normal(0, np.sqrt(sigma_sq))

    def sample(self, size: torch.Size, x: torch.Tensor):
        return self._inner_distr.rsample(size) + x

    def log_prob(self, samples, x):
        return self._inner_distr.log_prob(samples - x)


class ConditionalMultivariateNormal(NoiseDistr):
    """Multivariate normal distr. with adaptive mean"""

    def __init__(self, sigma_sq: float, dim=1):
        self.sigma_sq = sigma_sq
        self.dim = dim
        self._inner_distr = torch.distributions.MultivariateNormal(torch.zeros(self.dim),
                                                                   torch.eye(self.dim) * sigma_sq)

    def sample(self, size: torch.Size, x: torch.Tensor):
        assert x.size(-1) == self.dim

        x = torch.repeat_interleave(x, int(size[0] / x.size(0)), dim=0)
        assert x.size(0) == size[0]

        return self._inner_distr.rsample(size) + x

    def log_prob(self, samples, x):
        assert samples.size(-1) == self.dim and x.size(-1) == self.dim
        return self._inner_distr.log_prob(samples - x)

