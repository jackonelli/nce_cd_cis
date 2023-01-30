"""Adaptive proposal/noise distributions"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel
from src.models.gaussian_model import diag_check


class AdaptiveMdn(BaseModel):
    def __init__(self, num_components, input_dim, hidden_dim=10):
        super().__init__()

        self.num_components = num_components
        self.input_dim = input_dim

        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, self.num_components * input_dim)

        self.fc1_sigma = nn.Linear(input_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, self.num_components * input_dim)

        self.fc1_weight = nn.Linear(input_dim, hidden_dim)
        self.fc2_weight = nn.Linear(hidden_dim, self.num_components)

    def sample(self, size: torch.Size, x: Tensor) -> Tensor:
        """Sample from proposal distr.
        Args:
            size: (N, J), here N could also be a batch size B.
            x: Conditional value.
        """
        assert size[0] == x.size(0)
        # Ineffiecent to recompute the params for both sample and log_prob,
        # but it improves code structure

        means, sigmas, weights = self.predict_params(x)
        inds = torch.multinomial(weights, num_samples=size, replacement=True)
        # for ind in inds:
        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        y_samples_K = q_distr.sample(
            sample_shape=size
        )  # (shape: (num_samples, batch_size, K))
        y_samples = y_samples_K.gather(2, inds).squeeze(
            2
        )  # (shape: (num_samples, batch_size))
        return y_samples

    def log_prob(self, y: Tensor, x: Tensor) -> Tensor:
        """Compute log of unnorm prob: log p_tilde(y | x)"""
        means, sigmas, weights = self.predict_params(x)

        q_distr = torch.distributions.normal.Normal(loc=means, scale=sigmas)
        q_ys_K = torch.exp(
            q_distr.log_prob(torch.transpose(y, 1, 0).unsqueeze(2))
        )  # (shape: (1, batch_size, K))
        q_ys = torch.sum(
            weights.unsqueeze(0) * q_ys_K, dim=2
        )  # (shape: (1, batch_size))
        q_ys = q_ys.squeeze(0)  # (shape: (batch_size))
        return q_ys

    def predict_params(self, x):
        # (x_feature has shape: (batch_size, hidden_dim))

        means = F.relu(self.fc1_mean(x))  # (shape: (batch_size, hidden_dim))
        means = self.fc2_mean(means)  # (shape: batch_size, K))

        log_sigma2s = F.relu(self.fc1_sigma(x))  # (shape: (batch_size, hidden_dim))
        log_sigma2s = self.fc2_sigma(log_sigma2s)  # (shape: batch_size, K))
        sigma_sqs = torch.exp(log_sigma2s)  # (shape: (batch_size, K))

        weight_logits = F.relu(self.fc1_weight(x))  # (shape: (batch_size, hidden_dim))
        weight_logits = self.fc2_weight(weight_logits)  # (shape: batch_size, K))
        weights = torch.softmax(weight_logits, dim=1)  # (shape: batch_size, K))

        return means, sigma_sqs, weights


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
