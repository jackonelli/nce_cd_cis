"""Model predicting parameters of Gaussian distr."""
import torch
from torch import Tensor

from src.models.base_model import BaseModel


class DiagGaussianModel(BaseModel):
    def __init__(self, mu: Tensor, cov: Tensor):
        super().__init__()

        self.mu = torch.nn.Parameter(mu.clone(), requires_grad=True)
        diag_elements = torch.diagonal(cov.clone())
        assert torch.allclose(
            cov - torch.diag(diag_elements), torch.zeros(cov.size())
        ), "Expects diagonal cov. matrix"
        self._sqrt_diagonal = torch.nn.Parameter(
            torch.sqrt(diag_elements), requires_grad=True
        )

    def cov(self):
        return torch.diag(self._sqrt_diagonal ** 2)

    def log_prob(self, y: Tensor, x=None):
        return torch.distributions.MultivariateNormal(self.mu, self.cov()).log_prob(y)

    def sample(self, size: torch.Size, _x: torch.Tensor):
        inner_distr = torch.distributions.MultivariateNormal(self.mu, self.cov())
        return inner_distr.rsample(size)


class GaussianModel(BaseModel):
    def __init__(self, mu: Tensor, cov: Tensor):
        super().__init__()

        self.mu = torch.nn.Parameter(mu.clone(), requires_grad=True)
        self.cov = torch.nn.Parameter(cov.clone(), requires_grad=True)

    def log_prob(self, y: Tensor):
        return torch.distributions.MultivariateNormal(self.mu, self.cov).log_prob(y)
