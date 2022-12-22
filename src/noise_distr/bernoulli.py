import torch
from torch import Tensor
from src.noise_distr.base import NoiseDistr


class MultivariateBernoulli(NoiseDistr):
    def __init__(self, p: Tensor):

        self.p = p
        self._inner_distr = torch.distributions.Bernoulli(p)

    def sample(self, size: torch.Size, x=0):
        return self._inner_distr.sample(size)

    def log_prob(self, samples, x=0):
        return self._inner_distr.log_prob(samples).sum(dim=-1)


class ConditionalMultivariateBernoulli(NoiseDistr):
    def __init__(self, p_0: Tensor, p_1: Tensor):
        """Distribution with parameters p_0 if  conditioned on x=0, distribution with parameters p_1 otherwise"""

        self.p_0 = p_0
        self.p_1 = p_1
        self.dim = p_0.shape[0]

        self._inner_distr = torch.distributions.Bernoulli(p_0)
        self._inner_distr_1 = torch.distributions.Bernoulli(p_1)

    def sample(self, size: torch.Size, x: Tensor):

        return (1 - x) * self._inner_distr.sample(size) + x * self._inner_distr_1.sample(size)

    def log_prob(self, samples, x: Tensor):

        assert samples.size(-1) == self.dim
        return ((1 - x) * self._inner_distr.log_prob(samples) + x * self._inner_distr_1.log_prob(samples)).sum(dim=-1)