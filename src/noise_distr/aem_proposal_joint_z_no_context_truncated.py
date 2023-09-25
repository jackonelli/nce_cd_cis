# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

import torch
from torch.nn import functional as F
from torch import distributions

from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.experiments.aem_exp_utils import MixtureSameFamily1DNoR
from src.noise_distr.truncated_normal import TruncatedNormal


class AemJointProposalWOContextTruncated(AemJointProposal):
    def __init__(self, autoregressive_net, num_components, a, b,
                 mixture_component_min_scale=None, scale_activation=F.softplus, apply_context_activation=False):

        num_context_units = 0

        self.a, self.b = a, b

        super(AemJointProposalWOContextTruncated, self).__init__(autoregressive_net, num_context_units, num_components,
                                                                 mixture_component_min_scale, scale_activation,
                                                                 apply_context_activation)
        self.num_context_units = self.autoregressive_net.input_dim
        self.dim = self.a.shape[0]

    def forward(self, x):
        """Note: this is for one dimension"""
        params = self.autoregressive_net(x).reshape(-1, self.output_dim, self.made_output_dim_multiplier)

        num_obs = x[:, self.dim:].sum(dim=-1)
        a = torch.gather(self.a.reshape(1, -1).repeat(x.shape[0], 1), dim=-1, index=num_obs.reshape(-1, 1).long()).unsqueeze(dim=-1)
        b = torch.gather(self.b.reshape(1, -1).repeat(x.shape[0], 1), dim=-1, index=num_obs.reshape(-1, 1).long()).unsqueeze(dim=-1)

        # create proposal
        q = self.create_proposal_distr(params, a, b)

        return q, x  # Return x as "context"

    def create_proposal_distr(self, params, a, b):
        # separate out proposal params into coefficients, locs, and scales
        logits = params[..., :self.num_components]  # [B, D, M]
        if logits.shape[0] == 1:
            logits = logits.reshape(self.output_dim, self.num_components)
        locs = params[...,
               self.num_components:(
                       2 * self.num_components)]  # [B, D, M]
        scales = self.mixture_component_min_scale + self.scale_activation(
            params[..., (2 * self.num_components):])  # [B, D, M]

        # Force mean to be within boundaries
        #scales[scales < (a + 1)] = a[scales < (a + 1)] + 1
        #scales[scales > (b - 1)] = b[scales > (b - 1)] - 1

        mixture_distribution = distributions.OneHotCategorical(
            logits=logits,
            validate_args=True
        )

        components_distribution = TruncatedNormal(loc=locs, scale=scales, a=a, b=b)
        q = MixtureSameFamily1DNoR(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution
        )
        # del logits, locs, scales  # free GPU memory

        return q