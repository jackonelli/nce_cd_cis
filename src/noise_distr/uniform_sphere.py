import math
import torch
from src.noise_distr.base import NoiseDistr


class UniformSphere(NoiseDistr):

    def __init__(self, num_dims):
        self.num_dims = num_dims

        dims_factor = torch.tensor((self.num_dims - 1) / 2)
        self.log_surface = dims_factor * torch.log(torch.tensor(2 * math.pi)) - torch.lgamma(dims_factor)

    def sample(self, size: torch.Size, x=0):

        x_n = torch.randn((size[0], size[1], self.num_dims))
        return x_n / torch.sqrt(torch.sum(x_n**2, dim=-1, keepdim=True))

    def log_prob(self, samples, x=0):
        return - self.log_surface