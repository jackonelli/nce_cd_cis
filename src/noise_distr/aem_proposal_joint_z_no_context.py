# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

import torch
from torch.nn import functional as F

from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemJointProposalWOContext(AemJointProposal):
    def __init__(self, autoregressive_net, num_components,
                 mixture_component_min_scale=None, scale_activation=F.softplus, apply_context_activation=False):

        num_context_units = 0
        super(AemJointProposalWOContext, self).__init__(autoregressive_net, num_context_units, num_components,
                                                        mixture_component_min_scale, scale_activation,
                                                        apply_context_activation)
        self.num_context_units = self.autoregressive_net.input_dim

    def forward(self, x):
        """Note: this is for one dimension"""
        params = self.autoregressive_net(x).reshape(-1, self.output_dim, self.made_output_dim_multiplier)

        # create proposal
        q = self.create_proposal_distr(params)

        return q, x  # Return x as "context"

