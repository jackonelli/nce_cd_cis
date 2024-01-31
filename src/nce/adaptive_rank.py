"""Adaptive proposal distribution from ranking kernel

Re-use the ranking kernel to estimate the gradient of KL[p_theta || q_phi],
in order to jointly learn a parameterised proposal distribution q_phi.
"""

from typing import Optional
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator
from src.utils.part_fn_utils import norm_weights, unnorm_weights, concat_samples


class AdaptiveRankKernel(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def crit(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise(self._num_neg, y)

        return self.inner_crit(y, y_samples)

    def inner_crit(self, y: Tensor, y_samples: Tensor):

        w_tilde = self._unnorm_w(y, y_samples)
        return (-torch.log(w_tilde[:, 0]) + torch.log(w_tilde.sum(dim=1))).mean()

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):
        with torch.no_grad():
                y_samples = self.sample_noise(self._num_neg, y)
        return self.calculate_inner_crit_grad(y, y_samples)

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor):
        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()
        ys = concat_samples(y, y_samples)
        with torch.no_grad():
            weights = norm_weights(self._unnorm_w(y, y_samples))
        # Use negative weights since we want to estimate the negative grad log prob.
        grads = self._noise_distr.grad_log_prob(ys, -weights)
        # Assign calculated gradients to model parameters
        self._noise_distr.set_gradients(grads)

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (ranking version).

        Note that in the NCE ranking criterion we use a scaled version of Ẑ,
        L_NCE = -log(w_tilde(y_0)) + log ( (J+1) Ẑ),
        though it has no practical effect on the gradients.
        """

        w_tilde = self._unnorm_w(y, y_samples)
        return w_tilde.mean()

    def _unnorm_w(self, y, y_samples) -> Tensor:
        """Normalised weights of y (NxD) and y_samples (NxJxD)"""

        ys = concat_samples(y, y_samples)
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)
