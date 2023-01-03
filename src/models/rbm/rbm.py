"""Restricted Boltzmann Machine"""
import torch
from torch import Tensor

from src.models.base_model import BaseModel


# Adapted partly from https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
class Rbm(BaseModel):

    def __init__(self, weights, vis_bias, hidden_bias):
        super().__init__()

        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.vis_bias = torch.nn.Parameter(vis_bias, requires_grad=True)
        self.hidden_bias = torch.nn.Parameter(hidden_bias, requires_grad=True)

    def log_prob(self, y: Tensor) -> Tensor:
        return - self.energy(y)

    def energy(self, y: Tensor):
        # From http: // swoh.web.engr.illinois.edu / courses / IE598 / handout / rbm.pdf
        return - (torch.matmul(y, self.vis_bias) +
                  torch.log(1 + torch.exp(self.hidden_model(y))).sum(dim=-1, keepdim=True)).reshape(y.shape[0], -1)

    def total_energy(self, v: Tensor, h: Tensor):
        return - (torch.matmul(v, self.vis_bias) + torch.matmul(h, self.hidden_bias)
                  + (v * torch.matmul(h, self.weights.t())).sum(dim=-1, keepdim=True)).reshape(v.shape[0], -1)

    def hidden_model(self, y: Tensor):
        return torch.matmul(y, self.weights) + self.hidden_bias.t()

    def visible_model(self, h: Tensor):
        return torch.matmul(h, self.weights.t()) + self.vis_bias.t()

    def sample_hidden(self, y: Tensor):
        z = self.hidden_model(y)
        p_h = torch.sigmoid(z)
        sample_h = torch.distributions.bernoulli.Bernoulli(logits=z).sample()

        return p_h, sample_h

    def sample_visible(self, h: Tensor):
        z = self.visible_model(h)
        p_v = torch.sigmoid(z)
        sample_v = torch.distributions.bernoulli.Bernoulli(logits=z).sample()

        return p_v, sample_v

    def sample(self, y: Tensor, k=1):
        """Sample from distribution"""

        v = y.clone()
        for _ in range(k):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)

        return v


