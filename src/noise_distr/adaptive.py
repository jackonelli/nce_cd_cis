"""Adaptive proposal/noise distributions"""
import torch
from torch import Tensor
from src.models.base_model import BaseModel
from src.models.gaussian_model import diag_check


class AdaptiveDiagGaussianModel(BaseModel):
    """Adaptive noise distribution

    Specialised version of a diagonal MVN distribution,
    with learnable parameters.
    """

    # def __init__(self, mu: Tensor, cov: Tensor):
    #     super().__init__()

    #     self.mu = torch.nn.Parameter(mu.clone(), requires_grad=True)
    #     diag_elements = torch.diagonal(cov.clone())
    #     assert diag_check(
    #         cov
    #     ), f"{self.__class__.__name__} expects diagonal cov. matrix"

    #     self._sqrt_diagonal = torch.nn.Parameter(
    #         torch.sqrt(diag_elements), requires_grad=True
    #     )

    def __init__(self, mu: Tensor, cov: Tensor):
        super().__init__()
        self.mu = torch.nn.Parameter(mu.clone(), requires_grad=True)
        self.cov_ = torch.nn.Parameter(cov.clone(), requires_grad=True)

    def cov(self):
        # return torch.diag(self._sqrt_diagonal ** 2)
        return self.cov_

    # def cov(self):
    #     return torch.diag(self._sqrt_diagonal ** 2)

    def log_prob(self, y: Tensor, x=None):
        return torch.distributions.MultivariateNormal(self.mu, self.cov()).log_prob(y)

    def sample(self, size: torch.Size, _x: torch.Tensor):
        inner_distr = torch.distributions.MultivariateNormal(self.mu, self.cov())
        return inner_distr.rsample(size)
