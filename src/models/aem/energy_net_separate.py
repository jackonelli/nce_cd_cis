import torch
from torch import nn
from torch.nn import functional as F

from src.models.base_model import BaseModel
from src.models.aem.energy_net import ResidualBlock
from src.models.aem.made_joint_z import ResidualMADEJoint


class ResidualEnergyNetSep(BaseModel):
    def __init__(self, input_dim, made, n_residual_blocks=2, hidden_dim=32,
                 energy_upper_bound=None,
                 activation=F.relu, use_batch_norm=False, dropout_probability=None, zero_initialization=True,
                 apply_context_activation=False):
        super().__init__()

        self.activation = activation
        self.energy_upper_bound = energy_upper_bound
        self.apply_context_activation = apply_context_activation

        self.made = made

        self.dim = int(self.made.input_dim / 2)
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(
                features=hidden_dim,
                activation=activation,
                use_batch_norm=use_batch_norm,
                dropout_probability=0 if dropout_probability is None else dropout_probability,
                zero_initialization=zero_initialization
            )
            for _ in range(n_residual_blocks)
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):

        rel_inputs, cond_inputs = inputs[:, 0], inputs[:, 0:]
        del inputs  # free GPU memory

        print(cond_inputs.shape)
        context = self.made(cond_inputs)

        if self.apply_context_activation:
            context = self.activation(context)

        energy_net_inputs = torch.cat(
            (rel_inputs, context),
            dim=-1
        )

        del rel_inputs, context

        temps = self.initial_layer(energy_net_inputs)
        del energy_net_inputs

        for block in self.blocks:
            temps = block(temps)
        temps = self.activation(temps)
        outputs = self.final_layer(temps)
        if self.energy_upper_bound is not None:
            outputs = -F.softplus(outputs) + self.energy_upper_bound
        return outputs

