"""Restricted Boltzmann Machine"""
import torch
from torch import Tensor

from src.models.base_model import BaseModel


# Adapted from https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
# and https://heartbeat.comet.ml/guide-to-restricted-boltzmann-machines-using-pytorch-ee50d1ed21a8
# TODO: Skattar just nu energin med ett sample h. Hur löses detta vanligtvis?
class RBM(BaseModel):

    def __init__(self, weights, vis_bias, hidden_bias):
        super().__init__()

        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.vis_bias = torch.nn.Parameter(vis_bias, requires_grad=True)
        self.hidden_bias = torch.nn.Parameter(hidden_bias, requires_grad=True)

    def log_prob(self, y: Tensor) -> Tensor:
        _, h = self.sample_hidden(y)

        return torch.log(torch.exp(- self.energy(y, h)))

    def energy(self, v: Tensor, h: Tensor):
        return - torch.matmul(self.vis_bias.T, v) - torch.matmul(self.hidden_bias.T, h) \
               - torch.matmul(v.T, torch.matmul(self.weights, h))

    def sample_hidden(self, y: Tensor):
        p_h = torch.nn.functional.sigmoid(torch.nn.functional.linear(y, self.weights, self.hidden_bias))
        sample_h = torch.distributions.bernoulli.Bernoulli(p_h).sample()

        return p_h, sample_h

    def sample_visible(self, h: Tensor):
        p_v = torch.nn.functional.sigmoid(torch.nn.functional.linear(h, self.weights.T, self.vis_bias))
        sample_v = torch.distributions.bernoulli.Bernoulli(p_v).sample()

        return p_v, sample_v

    def sample(self, y: Tensor, k=1):
        """Sample from distribution"""

        v = y.clone()
        for _ in range(k):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)

        return v


