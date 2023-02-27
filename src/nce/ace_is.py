"""Noise Contrastive Estimation (NCE) ranking partition functions with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_unnorm_weights, concat_samples

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel
from src.experiments.ace_experiment_utils import UniformMaskGenerator


class AceIsCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        mcmc_steps: int,
        alpha: float = 0.0
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = mcmc_steps
        self.alpha = alpha  # For regularisation
        self.mask_generator = UniformMaskGenerator() # TODO: set seed?

    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        # Mask input
        y_o, y_u, observed_mask = self._mask_input(y)

        q, context = self._noise_distr.forward(y_o, observed_mask)
        y_samples = self.inner_sample_noise(q, num_samples=self._num_neg).detach().clone()

        return self.calculate_inner_crit_grad(q, context, (y_o, y_u, observed_mask), y_samples)

    def calculate_inner_crit_grad(self, q, context, y: tuple, y_samples: Tensor, y_base=None):

        # Mask input
        y_o, y_u, observed_mask = y

        # Calculate log prob for proposal
        log_noise_distr_y = self._noise_distr.inner_log_prob(q, y_u) * (1 - observed_mask)
        log_noise_distr_samples = self._noise_distr.inner_log_prob(q, y_samples.transpose(0, 1)).transpose(0, 1) \
                                  * (1 - observed_mask).unsqueeze(dim=1)

        q_mean = q.mean() * (1 - observed_mask)



        if y_base is None:
            # Gradient of mean is same as mean of gradient
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y)
        else:
            grads_log_prob_y = self._unnorm_distr.grad_log_prob(y_base)


        grads = [-grad_log_prob_y for grad_log_prob_y in grads_log_prob_y]
        y_0 = y.clone()
        for t in range(self.mcmc_steps):
            # Get neg. samples
            ys = concat_samples(y_0, y_samples)

            # Calculate and normalise weights
            w = self._norm_w(y_0, y_samples).detach()

            # Calculate gradients of log prob
            grads_log_prob = self._unnorm_distr.grad_log_prob(ys, w)

            # Sum over samples, mean over iter.
            grads = [
                grad + ((self._num_neg + 1) / self.mcmc_steps) * grad_log_prob
                for grad, grad_log_prob in zip(grads, grads_log_prob)
            ]

            if (t + 1) < self.mcmc_steps:
                # Sample y for next step
                sample_inds = torch.distributions.one_hot_categorical.OneHotCategorical(
                    probs=w
                ).sample()
                y_0 = ys[sample_inds.bool(), :]

                assert y_0.shape == y.shape

                # Sample neg. samples
                y_samples = self.sample_noise(self._num_neg, y_0)

                # TODO: resample mask?

        # Assign calculated gradients to model parameters
        self._unnorm_distr.set_gradients(grads)

    def sample_noise(self, num_samples: int, y: Tensor, q=None):
        return self._noise_distr.sample(torch.Size((num_samples,)), y)

    def inner_sample_noise(self, q, num_samples: int):
        return self._noise_distr.inner_sample(q, torch.Size((num_samples,)))

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _mask_input(self, y):
        observed_mask = self.mask_generator(y.shape[0], y.shape[-1])

        return y * observed_mask, y * (1 - observed_mask), observed_mask

    def _generate_model_input(self, y_u, y_samples, context, selected_features=None):

        # TODO: consider selected_features for inference
        ys_u = concat_samples(y_u, y_samples)

        u_i = torch.broadcast_to(torch.range(0, y_u.shape[-1], dtype=torch.int32),
            [y_u.shape[0], 1 + self._num_neg, y_u.shape[-1]],
        )

        if selected_features is not None:
            ys_u = ys_u[:, :, selected_features]# TODO: does this work as I think?
            u_i = u_i[:, :, selected_features]
            context = context[:, selected_features]

        tiled_context = context.unsqueeze(dim=1)