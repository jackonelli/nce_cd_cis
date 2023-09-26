import torch
from torch import nn
from torch.nn import functional as F, init

from src.models.base_model import BaseModel


def get_mask(in_features, out_features, autoregressive_features, mask_type=None):
    max_ = max(1, autoregressive_features - 1)
    min_ = min(1, autoregressive_features - 1)

    if mask_type == 'input':
        in_degrees = torch.arange(1, autoregressive_features + 1)
        out_degrees = torch.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees).float()

    elif mask_type == 'output':
        in_degrees = torch.arange(in_features) % max_ + min_
        out_degrees = tile(
            torch.arange(1, autoregressive_features + 1),
            out_features // autoregressive_features
        )
        mask = (out_degrees[..., None] > in_degrees).float()

    else:
        in_degrees = torch.arange(in_features) % max_ + min_
        out_degrees = torch.arange(out_features) % max_ + min_
        mask = (out_degrees[..., None] >= in_degrees).float()

    return mask


def get_autoregressive_mask(dim):
    return torch.cat((torch.zeros(1, dim), get_mask(1, dim - 1, dim, 'input'), torch.ones(1, dim)))


class ResidualBlock(nn.Module):
    def __init__(self, features, activation=F.relu,
                 zero_initialization=True, dropout_probability=0., use_batch_norm=False):
        super().__init__()
        self.features = features
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([
                nn.BatchNorm1d(features, eps=1e-3)
                for _ in range(2)
            ])
        self.layers = nn.ModuleList([
            nn.Linear(features, features)
            for _ in range(2)
        ])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.zeros_(self.layers[-1].weight)
            init.zeros_(self.layers[-1].bias)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.layers[1](temps)
        return temps + inputs


class MADEJoint(BaseModel):
    def __init__(self, input_dim, n_hidden_layers, hidden_dim, output_dim_multiplier,
                 conditional=False, conditioning_dim=None, activation=F.relu):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = 1
        self.output_dim_multiplier = output_dim_multiplier
        self.conditional = conditional

        self.initial_layer = nn.Linear(
            input_dim,
            hidden_dim,
        )
        if conditional:
            assert conditioning_dim is not None, 'Dimension of condition variables must be specified.'
            self.conditional_layer = nn.Linear(conditioning_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim)
             for _ in range(n_hidden_layers)]
        )
        self.final_layer = nn.Linear(
            hidden_dim, self.output_dim * output_dim_multiplier
        )

        self.activation = activation

    def forward(self, inputs, conditional_inputs=None):
        temps = self.initial_layer(inputs)
        if self.conditional:
            temps += self.conditional_layer(conditional_inputs)
        temps = self.activation(temps)
        for layer in self.hidden_layers:
            temps = layer(temps)
            temps = self.activation(temps)
        outputs = self.final_layer(temps)
        return outputs


class ResidualMADEJoint(BaseModel):
    def __init__(self, input_dim, n_residual_blocks, hidden_dim,
                 output_dim_multiplier, conditional=False, conditioning_dim=None,
                 activation=F.relu, use_batch_norm=False,
                 dropout_probability=None, zero_initialization=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.output_dim_multiplier = output_dim_multiplier
        self.conditional = conditional

        self.initial_layer = nn.Linear(
            input_dim,
            hidden_dim,
        )
        if conditional:
            assert conditioning_dim is not None, 'Dimension of condition variables must be specified.'
            self.conditional_layer = nn.Linear(conditioning_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(
                features=hidden_dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_probability=0 if dropout_probability is None else dropout_probability,
                zero_initialization=zero_initialization
            )
                for _ in range(n_residual_blocks)]
        )
        self.final_layer = nn.Linear(
            hidden_dim,
            self.output_dim * output_dim_multiplier
        )

        self.activation = activation

    def forward(self, inputs, conditional_inputs=None):
        temps = self.initial_layer(inputs)
        del inputs  # free GPU memory
        if self.conditional:
            temps += self.conditional_layer(conditional_inputs)
        for block in self.blocks:
            temps = block(temps)
        temps = self.activation(temps)

        outputs = self.final_layer(temps)
        return outputs




