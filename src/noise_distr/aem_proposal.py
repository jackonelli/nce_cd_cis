# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines

import torch
from torch import distributions
from torch.nn import functional as F

from src.models.base_model import BaseModel

from src.experiments.aem_exp_utils import Normal_, MixtureSameFamily


class AemProposal(BaseModel):
    def __init__(self, autoregressive_net, proposal_component_family, num_context_units, num_components,
                 mixture_component_min_scale=None, scale_activation=F.softplus, apply_context_activation=False):

        super(AemProposal, self).__init__()

        self.autoregressive_net = autoregressive_net
        self.dim = self.autoregressive_net.input_dim

        if proposal_component_family == 'gaussian':
            self.Component = Normal_
        elif proposal_component_family == 'cauchy':
            self.Component = distributions.Cauchy
        elif proposal_component_family == 'laplace':
            self.Component = distributions.Laplace
        elif proposal_component_family == 'uniform':
            self.Component = None

        self.num_components = num_components  # M
        self.num_context_units = num_context_units
        self.made_output_dim_multiplier = num_context_units + 3 * num_components
        self.mixture_component_min_scale = 0 if mixture_component_min_scale is None else mixture_component_min_scale
        self.scale_activation = scale_activation
        self.apply_context_activation = apply_context_activation

    def _sample_batch(self, batch_size, return_log_density_of_samples=False):
        # need to do n_samples passes through autoregressive net
        samples = torch.zeros(batch_size, self.autoregressive_net.input_dim)
        log_density_of_samples = torch.zeros(batch_size,
                                             self.autoregressive_net.input_dim)
        for dim in range(self.autoregressive_net.input_dim):
            # compute autoregressive outputs
            autoregressive_outputs = self.autoregressive_net(samples).reshape(-1, self.dim, self.autoregressive_net.output_dim_multiplier)

            # grab proposal params for dth dimensions
            params = autoregressive_outputs[..., dim, self.num_context_units:]

            # make mixture coefficients, locs, and scales for proposal
            logits = params[...,
                     :self.num_components]  # [B, D, M]
            if logits.shape[0] == 1:
                logits = logits.reshape(self.dim, self.num_components)
            locs = params[...,
                   self.num_components:(
                           2 * self.num_components)]  # [B, D, M]
            scales = self.mixture_component_min_scale + self.scale_activation(
                params[...,
                (2 * self.num_components):])  # [B, D, M]

            # create proposal
            if self.Component is not None:
                mixture_distribution = distributions.OneHotCategorical(
                    logits=logits,
                    validate_args=True
                )
                components_distribution = self.Component(loc=locs, scale=scales)
                self.proposal = MixtureSameFamily(
                    mixture_distribution=mixture_distribution,
                    components_distribution=components_distribution
                )
                proposal_samples = self.proposal.sample((1,))  # [S, B, D]

            else:
                self.proposal = distributions.Uniform(low=-4, high=4)
                proposal_samples = self.proposal.sample(
                    (1, batch_size, 1)
                )
            proposal_samples = proposal_samples.permute(1, 2, 0)  # [B, D, S]
            proposal_log_density = self.proposal.log_prob(proposal_samples)
            log_density_of_samples[:, dim] += proposal_log_density.reshape(-1).detach()
            samples[:, dim] += proposal_samples.reshape(-1).detach()

        if return_log_density_of_samples:
            return samples, torch.sum(log_density_of_samples, dim=-1)
        else:
            return samples

    def sample_over_batches(self, n_samples, return_log_density_of_samples=False, batch_size=10000):
        if n_samples > batch_size:
            # determine how many batches are needed
            n_batches, leftover = n_samples // batch_size, n_samples % batch_size

            # get batches
            samples = torch.zeros(n_samples, self.autoregressive_net.input_dim)
            log_density_of_samples = torch.zeros(n_samples)
            for n in range(n_batches):
                batch_of_samples, log_density_of_batch_of_samples = self._sample_batch(
                    batch_size, return_log_density_of_samples=True)
                index = slice((batch_size * n), (batch_size * (n + 1)))
                samples[index, :] += batch_of_samples
                log_density_of_samples[index] += log_density_of_batch_of_samples

            if leftover:
                batch_of_samples, log_density_of_batch_of_samples = self._sample_batch(
                    leftover, return_log_density_of_samples=True)
                samples[-leftover:, :] += batch_of_samples
                log_density_of_samples[-leftover:, :] += log_density_of_batch_of_samples

            if return_log_density_of_samples:
                return samples, log_density_of_samples
            else:
                return samples

        else:
            return self._sample_batch_from_proposal(n_samples,
                                                    return_log_density_of_samples)

    def sample(self, size: torch.Size, x):
        # Note: expect x to be a tuple (observed input, observed_mask)

        proposal_distr, _ = self.forward(x)

        return self.inner_sample(proposal_distr, size)  # TODO: Need to check so that size is correct

    def inner_sample(self, distr, size):
        return distr.sample(size).transpose(0, 1)

    def forward(self, x, conditional_inputs=None):

        if conditional_inputs is not None:
            autoregressive_outputs = self.autoregressive_net(x, conditional_inputs).reshape(-1, self.dim,
                                                                                            self.made_output_dim_multiplier)
        else:
            autoregressive_outputs = self.autoregressive_net(x).reshape(-1, self.dim, self.made_output_dim_multiplier)

        context, params = (
            autoregressive_outputs[..., :self.num_context_units],
            autoregressive_outputs[..., self.num_context_units:]
        )

        del autoregressive_outputs  # free GPU memory
        if self.apply_context_activation:
            context = self._unnorm_distr.activation(context)

        # create proposal
        q = self.create_proposal_distr(params)

        return q, context

    def forward_along_dim(self, x, dim):

        autoregressive_outputs = self.autoregressive_net(x).reshape(-1, self.dim, self.made_output_dim_multiplier)

        context, params = (
            autoregressive_outputs[..., dim, :self.num_context_units][:, None, :],
            autoregressive_outputs[..., dim, self.num_context_units:][:, None, :]
        )

        del autoregressive_outputs  # free GPU memory
        if self.apply_context_activation:
            context = self._unnorm_distr.activation(context)

        # separate out proposal params into coefficients, locs, and scales
        # create proposal
        q = self.create_proposal_distr(params)

        return q, context

    def get_context(self, inputs):
        # get energy params and proposal params
        autoregressive_outputs = self.autoregressive_net(inputs).reshape(-1, self.dim,
                                                                         self.made_output_dim_multiplier)
        context_params = autoregressive_outputs[..., :self.num_context_units]
        return context_params



    def create_proposal_distr(self, params):

        # separate out proposal params into coefficients, locs, and scales
        logits = params[..., :self.num_components]  # [B, D, M]
        if logits.shape[0] == 1:
            logits = logits.reshape(self.dim, self.num_components)
        locs = params[...,
               self.num_components:(
                       2 * self.num_components)]  # [B, D, M]
        scales = self.mixture_component_min_scale + self.scale_activation(
            params[..., (2 * self.num_components):])  # [B, D, M]

        q = None
        if self.Component is not None:
            mixture_distribution = distributions.OneHotCategorical(
                logits=logits,
                validate_args=True
            )
            components_distribution = self.Component(loc=locs, scale=scales)
            q = MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                components_distribution=components_distribution
            )
        else:
            q = distributions.Uniform(low=-4, high=4)

        #del logits, locs, scales  # free GPU memory

        return q

    def log_prob(self, samples, x=0):
        # Calculate log prob conditionen on x
        proposal_distr, _ = self.forward(samples)

        return self.inner_log_prob(proposal_distr, samples)

    def inner_log_prob(self, distr, samples):
        return distr.log_prob(samples)

    def prob(self, samples):
        """Probability of a sample y"""
        return torch.exp(self.log_prob(samples))

    def get_autoregressive_net(self):
        return self.autoregressive_net
