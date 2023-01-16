import torch
from torch import Tensor

from src.models.base_model import BaseModel

from src.training.training_utils import no_change_stopping_condition


class CondEbm(BaseModel):
    def __init__(self, input_weights, input_bias, hidden_weights, hidden_bias, input_class_weights, input_class_bias,
                 hidden_class_weights, hidden_class_bias):

        super().__init__()

        self.input_weights = torch.nn.Parameter(input_weights, requires_grad=True)
        self.input_bias = torch.nn.Parameter(input_bias, requires_grad=True)
        self.hidden_weights = torch.nn.Parameter(hidden_weights, requires_grad=True)
        self.hidden_bias = torch.nn.Parameter(hidden_bias, requires_grad=True)

        self.input_class_weights = torch.nn.Parameter(input_class_weights, requires_grad=True)
        self.input_class_bias = torch.nn.Parameter(input_class_bias, requires_grad=True)
        self.hidden_class_weights = torch.nn.Parameter(hidden_class_weights, requires_grad=True)
        self.hidden_class_bias = torch.nn.Parameter(hidden_class_bias, requires_grad=True)

    def energy(self, y: Tensor, x: Tensor) -> Tensor:
        """Compute energy"""

        y_concat = y + torch.matmul(x, self.input_class_bias)
        z = torch.nn.functional.leaky_relu(torch.matmul(x, self.input_class_weights)
                                           * torch.matmul(y_concat, self.input_weights) + torch.matmul(x, self.input_bias))

        z_concat = z + torch.matmul(x, self.hidden_class_bias)
        return (torch.matmul(x, self.hidden_class_weights) * torch.matmul(z_concat, self.hidden_weights) + torch.matmul(x, self.hidden_bias)).squeeze(dim=-1)

    def log_prob(self, y: tuple) -> Tensor:
        y, x = y
        return - self.energy(y, x)

    def sample(self, num_samples=1, lr=0.1, num_epochs=100):

        y_samples = []
        class_distr = torch.distributions.one_hot_categorical.OneHotCategorical(torch.tensor([1 / self.input_bias.shape[0]] * self.input_bias.shape[0]))

        for i in range(num_samples):
            # TODO: Freeze model parameters?
            y_init = torch.randn(1, self.input_weights.shape[0])
            y = torch.nn.Parameter(y_init, requires_grad=True)
            x = class_distr.sample((1,))

            print("Sampling from class {}".format(x.argmax()))

            optimizer = torch.optim.SGD([y], lr=lr)

            old_y = torch.nn.utils.parameters_to_vector(y)
            for epoch in range(num_epochs):
                optimizer.zero_grad()

                loss = - self.log_prob((torch.sigmoid(y), x))  #y / torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True)))
                loss.backward()

                optimizer.step()

                #if no_change_stopping_condition(
                #        torch.nn.utils.parameters_to_vector(y), old_y
                #):
                #    print("Training converged in epoch {}".format(epoch))
                #    break

                old_y = torch.nn.utils.parameters_to_vector(y)

            y_samples.append(torch.sigmoid(y).detach().clone())  #/ torch.sqrt(torch.sum(y**2, dim=-1, keepdim=True))).detach().clone())

        return torch.cat(y_samples, dim=0)

    def concat_data(self, y, x):
        if (y.ndim == 2) or (y.ndim == 3 and y.shape[1] == x.shape[1]):
            return torch.concat((y, x), dim=-1)
        else:
            return torch.concat((y, x.repeat(1, y.shape[1], 1)), dim=-1)

