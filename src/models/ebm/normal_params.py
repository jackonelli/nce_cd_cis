"""Normal (Gaussian) distribution parameterised as an EBM.

Toy model to showcase the EBM class. The parameters are

    theta = [mu sigma^2]

with energy

    E_theta(y) = (y - mu)^2 / (2 sigma^2)

With the exponential function as the positive mapping the full model becomes

    p_tilde_theta(y) = exp( - E_theta(y) ) = exp( - (y - mu)^2 / (2 sigma^2) )

which is an unnormalised normal pdf.
"""
from math import pi
import torch
from src.models.ebm.base import EbmBase


class NormalEbm(EbmBase):
    def __init__(self, mu, sigma_sq):
        super().__init__(torch.exp)
        self.mu = mu
        self.sigma_sq = sigma_sq

    def energy(self, y, x=None):
        dist_sq = (y - self.mu) ** 2
        return dist_sq / (2 * self.sigma_sq)

    def true_part_fn(self):
        """Exact normalisation for the normal EBM"""
        return normal_part_fn(self.sigma_sq)

    def log_prob(self, y):
        """Compute energy E_theta(y, x)

        This method overrides the general log_prob method of the EBM base class
        since we for this toy model have access to an exact normalisation.
        """
        return -self.energy(y) - torch.log(self.true_part_fn())


def normal_part_fn(sigma_sq):
    """Compute exact part fn for a normal distr. with variance sigma_sq"""
    return torch.sqrt(2 * pi * sigma_sq)
