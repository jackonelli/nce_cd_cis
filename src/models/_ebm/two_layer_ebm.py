import torch
from torch import Tensor

from src.models.base_model import BaseModel

from src.training.training_utils import no_change_stopping_condition


class Ebm(BaseModel):
    def __init__(self, input_weights, input_bias, hidden_weights, hidden_bias):

        super().__init__()

        self.input_weights = torch.nn.Parameter(input_weights, requires_grad=True)
        self.input_bias = torch.nn.Parameter(input_bias, requires_grad=True)
        self.hidden_weights = torch.nn.Parameter(hidden_weights, requires_grad=True)
        self.hidden_bias = torch.nn.Parameter(hidden_bias, requires_grad=True)

    def energy(self, y: Tensor) -> Tensor:
        """Compute energy"""

        z = torch.matmul(y, self.input_weights) + self.input_bias.t()

        return (torch.matmul(z, self.hidden_weights) + self.hidden_bias.t()).squeeze(dim=-1)

    def log_prob(self, y: Tensor) -> Tensor:
        return - self.energy(y)

    def sample(self, num_samples=1, lr=0.1, num_epochs=100):
        y_samples = []

        for i in range(num_samples):
            # TODO: Freeze model parameters?
            y_init = torch.randn(1, self.input_weights.shape[0])
            y = torch.nn.Parameter(y_init, requires_grad=True)
            optimizer = torch.optim.SGD([y], lr=lr)

            old_y = torch.nn.utils.parameters_to_vector(y)
            for epoch in range(num_epochs):
                optimizer.zero_grad()

                loss = - self.log_prob(torch.sigmoid(y))  #y / torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True)))
                loss.backward()

                optimizer.step()

                if no_change_stopping_condition(
                        torch.nn.utils.parameters_to_vector(y), old_y
                ):
                    print("Training converged")
                    break

            y_samples.append(torch.sigmoid(y).detach().clone())  #/ torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True))).detach().clone())

        return torch.cat(y_samples, dim=0)