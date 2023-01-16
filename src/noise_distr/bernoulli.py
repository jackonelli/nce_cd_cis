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

