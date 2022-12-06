"""Model predicting parameters of Gaussian distr."""
import torch
from torch import Tensor

from src.models.base_model import BaseModel


class GaussianModel(BaseModel):
    def __init__(self, mu: Tensor, cov: Tensor):
        super().__init__()

        self.mu = torch.nn.Parameter(mu, requires_grad=True)
        self.cov = torch.nn.Parameter(cov, requires_grad=True)

    def log_prob(self, y: Tensor):
        return torch.distributions.MultivariateNormal(self.mu, self.cov).log_prob(y)