"""Ring model"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor


class RingModel(torch.nn.Module):
    def __init__(self, mu, log_precision):
        super().__init__()
        self.mu = mu
        self.log_precision = torch.nn.Parameter(log_precision, requires_grad=True)

    def unnorm_prob(self, y: Tensor) -> Tensor:
        """Compute unnorm prob p_tilde(y)"""

        return torch.exp(self.log_prob(y))

    def log_prob(self, y: Tensor):
        return ring_model_pdf(y, self.mu, torch.exp(self.log_precision))

    def forward(self, y):
        return self.log_prob(y)


class RingModelNCE(torch.nn.Module):
    def __init__(self, mu, log_precision, log_part_fn):
        super().__init__()

        self.mu = mu
        self.log_precision = torch.nn.Parameter(log_precision, requires_grad=True)
        self.log_part_fn = torch.nn.Parameter(log_part_fn, requires_grad=True)

    def unnorm_prob(self, y: Tensor) -> Tensor:
        """Compute unnorm prob p_tilde(y)"""

        return torch.exp(self.log_prob(y))

    def log_prob(self, y: Tensor):
        return ring_model_pdf(y, self.mu, torch.exp(self.log_precision)) + self.log_part_fn

    def forward(self, y):
        return self.log_prob(y)


def ring_model_pdf(x, mu, precision):
    return - (precision / 2) * (torch.norm(x, p=2, dim=-1) - mu)**2


def plot_ring_model_pdf():
    nx = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(nx, nx)
    x = torch.tensor(np.column_stack((X.reshape(-1), Y.reshape(-1))))

    log_pdf = ring_model_pdf(x, mu=2, precision=1).reshape(100, 100)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    ax.contourf(X, Y, log_pdf, cmap='ocean')
    plt.show()


if __name__ == '__main__':
    plot_ring_model_pdf()