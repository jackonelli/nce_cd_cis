# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

import torch
from torch import distributions
from torch.nn import functional as F

from src.models.base_model import BaseModel

from src.experiments.aem_exp_utils import Normal_, MixtureSameFamily1D


class AemJointProposal(BaseModel):
    def __init__(self, autoregressive_net, num_context_units, num_components,
                 mixture_component_min_scale=None, scale_activation=F.softplus, apply_context_activation=False):

        super(AemJointProposal, self).__init__()

        self.autoregressive_net = autoregressive_net
        self.num_features = int(self.autoregressive_net.input_dim / 2)
        self.output_dim = self.autoregressive_net.output_dim

        self.num_components = num_components  # M
        self.num_context_units = num_context_units
        self.made_output_dim_multiplier = num_context_units + 3 * num_components
        self.mixture_component_min_scale = 0 if mixture_component_min_scale is None else mixture_component_min_scale
        self.scale_activation = scale_activation
        self.apply_context_activation = apply_context_activation

    def sample(self, size: torch.Size, x):
        # Note: expect x to be a tuple (observed input, observed_mask)

        proposal_distr, _ = self.forward(x)

        return self.inner_sample(proposal_distr, size)  # TODO: Need to check so that size is correct

    def inner_sample(self, distr, size):
        return distr.sample(size).transpose(0, 1)

    def forward(self, x):
        """Note: this is for one dimension"""

        autoregressive_outputs = self.autoregressive_net(x).reshape(-1, self.output_dim, self.made_output_dim_multiplier)

        context, params = (
            autoregressive_outputs[..., :self.num_context_units],
            autoregressive_outputs[..., self.num_context_units:]
        )

        del autoregressive_outputs  # free GPU memory
        if self.apply_context_activation:
            context = self._unnorm_distr.activation(context)

        # create proposal
        q = self.create_proposal_distr(params)

        return q, context.squeeze()

    def get_context(self, inputs):
        # get energy params and proposal params
        autoregressive_outputs = self.autoregressive_net(inputs).reshape(-1, self.output_dim,
                                                                         self.made_output_dim_multiplier)
        context_params = autoregressive_outputs[..., :self.num_context_units]
        return context_params

    def create_proposal_distr(self, params):

        # separate out proposal params into coefficients, locs, and scales
        logits = params[..., :self.num_components]  # [B, D, M]
        if logits.shape[0] == 1:
            logits = logits.reshape(self.output_dim, self.num_components)
        locs = params[...,
               self.num_components:(
                       2 * self.num_components)]  # [B, D, M]
        scales = self.mixture_component_min_scale + self.scale_activation(
            params[..., (2 * self.num_components):])  # [B, D, M]

        mixture_distribution = distributions.OneHotCategorical(
            logits=logits,
            validate_args=True
        )
        components_distribution = Normal_(loc=locs, scale=scales)
        q = MixtureSameFamily1D(
            mixture_distribution=mixture_distribution,
            components_distribution=components_distribution
        )
        #del logits, locs, scales  # free GPU memory

        return q

    def log_prob(self, samples, x=0):
        # Calculate log prob conditionen on x
        proposal_distr, _ = self.forward(x)

        return self.inner_log_prob(proposal_distr, samples)

    def inner_log_prob(self, distr, samples):
        return distr.log_prob(samples)

    def prob(self, samples):
        """Probability of a sample y"""
        return torch.exp(self.log_prob(samples))

    def get_autoregressive_net(self):
        return self.autoregressive_net
