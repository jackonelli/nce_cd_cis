import math
import torch
from torch import Tensor
from src.noise_distr.base import NoiseDistr

# Skulle vilja ha det här: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution, men det verkar krångligt att sampla ifrån

class ConditionalSphere(NoiseDistr):

    def __init__(self, num_dims, kappa=1.0):

        self.num_dims = num_dims
        self.kappa = kappa

        self._inner_distr = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((num_dims,)),
                                                                                       (1 / kappa) * torch.eye(num_dims))

        dims_factor = torch.tensor((self.num_dims - 1) / 2)
        self.log_surface = dims_factor * torch.log(torch.tensor(2 * math.pi)) - torch.lgamma(dims_factor)

        dims_factor = torch.tensor((self.num_dims - 1) / 2)
        self.log_surface = dims_factor * torch.log(torch.tensor(2 * math.pi)) - torch.lgamma(dims_factor)

    def sample(self, size: torch.Size, x: Tensor):

        x_n = self._inner_distr.sample(size) + x

        return x_n / torch.sqrt(torch.sum(x_n ** 2, dim=-1, keepdim=True))

    def log_prob(self, samples, x: Tensor):
        return - self.log_surface

