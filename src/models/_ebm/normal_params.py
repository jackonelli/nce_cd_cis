"""Toy example model with a specific energy based model (EBM)
for estimating parameters of a normal distribution.

The parameters are

    theta = [mu sigma^2]

with energy

    E_theta(y) = (y - mu)^2 / (2 sigma^2)

With the exponential function as the positive mapping the full model becomes

    s(y) = exp( - E_theta(y) ) = exp( - (y - mu)^2 / (2 sigma^2) )

which is an unnormalised normal pdf.
"""
from math import pi
import torch
from src.models._ebm.base import Ebm


class NormalEbm(Ebm):
    def __init__(self, mu, sigma_sq):
        super().__init__(torch.exp)
        self.mu = mu
        self.sigma_sq = sigma_sq

    def energy(self, y):
        dist_sq = (y - self.mu) ** 2
        return dist_sq / (2 * self.sigma_sq)

    def true_part_fn(self):
        return normal_part_fn(self.sigma_sq)

    def prob(self, y):
        return self.unnorm_prob(y) / self.true_part_fn()

    def log_prob(self, y):
        return -self.energy(y) - torch.log(self.true_part_fn())


def normal_part_fn(sigma_sq):
    """Compute exact part fn for a normal distr. with variance sigma_sq"""
    return torch.sqrt(2 * pi * sigma_sq)

