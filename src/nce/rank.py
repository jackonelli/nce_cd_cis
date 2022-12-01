"""Noise Contrastive Estimation (NCE) partition functions"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator, unnorm_weights


class NceRankCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr):

        super().__init__(unnorm_distr, noise_distr)

    def crit(self, y: Tensor, y_samples: Tensor) -> Tensor:

        w_tilde = self._unnorm_w(y, y_samples)

        w_norm = torch.cat((w_tilde[:y.shape[0]].reshape(-1, 1), w_tilde[y.shape[0]:].reshape(y.shape[0], -1)), dim=-1)
        assert w_norm.shape[-1] == (y_samples.shape[0] / y.shape[0] + 1)

        return (- torch.log(w_tilde[:y.shape[0]]) + torch.log(torch.sum(w_norm, dim=-1))).mean()

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


