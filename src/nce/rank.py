"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights, concat_samples


class NceRankCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, num_neg_samples):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

    def crit(self, y: Tensor) -> Tensor:
        y_samples = self.sample_noise(self._num_neg * y.size(0), y)

        return self.inner_crit(y, y_samples)

    def inner_crit(self, y: Tensor, y_samples: Tensor):

        w_tilde = self._unnorm_w(y, y_samples)

        w_concat = concat_samples(w_tilde[:y.shape[0]], w_tilde[y.shape[0]:])
        assert w_concat.shape[1] == (y_samples.shape[0] / y.shape[0] + 1)

        return (- torch.log(w_tilde[:y.shape[0]]) + torch.log(w_concat.sum(dim=1))).mean()

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with NCE (ranking version).

        Note that in the NCE ranking criterion we use a scaled version of Ẑ,
        L_NCE = -log(w_tilde(y_0)) + log ( (J+1) Ẑ),
        though it has no practical effect on the gradients.
        """

        w_tilde = self._unnorm_w(y, y_samples)
        return w_tilde.mean()

    def _unnorm_w(self, y, y_samples) -> Tensor:

        if y.ndim == 1:
            y = y.reshape((1, -1))

        ys = torch.cat((y, y_samples))
        return unnorm_weights(ys, self._unnorm_distr.prob, self._noise_distr.prob)

