# Arbitrary Conditional Distributions with Energy (https://github.com/lupalab/ace/blob/main/ace/networks.py)
import torch

from src.models.base_model import BaseModel


class AceModel(BaseModel):

    def __init__(self, num_features: int, num_context_units: int, num_blocks: int = 4, num_hidden_units: int = 128,
                 activation: str = "relu", dropout_rate: float = 0.0, energy_clip: float = 30.0):

        super(AceModel, self).__init__()

        self.num_features = num_features
        self.input_dim = num_features + num_context_units + 1

        # TODO: handle this in nicer way (or always use relu)
        if activation == "relu":
            self.activation_fun = torch.nn.ReLU
        else:
            print("unknown activation")
            self.activation_fun = torch.nn.ReLU

        self.input_layer = torch.nn.Linear(in_features=self.input_dim, out_features=num_hidden_units)

        self.num_blocks = num_blocks
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(num_hidden_units, num_hidden_units, self.activation_fun,
                                                                  dropout_rate) for _ in range(num_blocks)])

        self.output_layer = torch.nn.Linear(num_hidden_units, 1)
        self.energy_clip = energy_clip

    def energy(self, y: tuple):
        # Note: expect y to be a tuple (masked input (unobserved set to 0), index of unobserved input, context (from proposal net))

        x_u_i, u_i, context = y
        u_i_one_hot = torch.nn.functional.one_hot(u_i, self.num_features)

        h = torch.cat((x_u_i.unsqueeze(dim=-1), u_i_one_hot, context), dim=-1).type(torch.float32)
        x = self.activation_fun()(self.input_layer(h))

        for module in self.residual_blocks:
            x = module(x)

        return torch.nn.Softplus()(self.output_layer(x))

    def log_prob(self, y: tuple):
        return - torch.clip(self.energy(y), 0.0, self.energy_clip)


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, activation_fun, dropout_rate: float):
        super(ResidualBlock, self).__init__()

        self.activation_fun = activation_fun

        self.input_layer = torch.nn.Linear(input_dim, hidden_units)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.output_layer = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, x):

        y = self.activation_fun()(self.input_layer(x))
        y = self.dropout(y)
        y = self.activation_fun()(x + self.output_layer(y))

        return y
