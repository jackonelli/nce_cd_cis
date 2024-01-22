"""Persistent Contrastive divergence (CD) with Gibbs sampling (for RBMs) """

from typing import Optional
import torch
from torch import Tensor

from src.models.rbm.rbm import Rbm
from src.noise_distr.base import NoiseDistr
from src.cd.cd_gibbs import CdGibbsCrit


class PersistentCdGibbsCritBatch(CdGibbsCrit):
    """Persistent CD crit with Gibbs sampling"""

    def __init__(self, unnorm_distr: Rbm, noise_distr: NoiseDistr, num_neg_samples: int):
        mcmc_steps = 1  #TODO: If we want to take several MCMC-steps, persistent y should  be updated at end of gradient calculation?
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, mcmc_steps)
        self._persistent_y = dict()
        self._eps = 0.05
        self._eps_distr = torch.distributions.bernoulli.Bernoulli(probs=self._eps)

    def calculate_crit_grad(self, y: Tensor, idx: Optional[Tensor]):
        assert (
            idx is not None
        ), "PersistentCdGibbsCrit requires an idx tensor that is not None"

        # TODO: this is a bit of an override, and should be made nice if we keep it
        idx = torch.arange(start=0, end=y.shape[0])

        y_p = self.persistent_y(y, idx)
        h, y_sample, h_sample = self.gibbs_sample(y_p)

        self._update_persistent_y(y_p, idx)

        return self.calculate_inner_crit_grad((y_p, h), (y_sample, h_sample))

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ with CD
        """
        # TODO: idx parameter
        return super().part_fn(y, y_samples)

    # def persistent_y(self, actual_y: Tensor, idx: Tensor):
    #     """Get persistent y
    #
    #     Access the last selected y or return the y sampled from p_d
    #     if no last selected y exists
    #     """
    #     per_y = torch.empty(actual_y.size())
    #     for n, per_n in enumerate(idx):
    #         per_n = per_n.item()
    #         per_y[n, :] = (
    #             self._persistent_y[per_n]
    #             if self._persistent_y.get(per_n) is not None
    #             else actual_y[n, :]
    #         )
    #     return per_y

    def persistent_y(self, actual_y: Tensor, idx: Tensor):
        """Get persistent y

        Access the last selected y or return the y sampled from p_d
        if no last selected y exists
        """

        per_y = torch.empty(actual_y.size())
        for n, per_n in enumerate(idx):
            per_n = per_n.item()
            per_y[n, :] = (
                self._sample_pers(actual_y[n, :], self._persistent_y[per_n])
                if self._persistent_y.get(per_n) is not None
                else actual_y[n, :]
            )
        return per_y

    def _sample_pers(self, y, y_p):
        sample_inds = self._eps_distr.sample((1,))
        return sample_inds * y + (1 - sample_inds) * y_p

    def _update_persistent_y(self, y, idx):
        """Assign new persistent y"""
        for n, _ in enumerate(y):
            self._persistent_y[idx[n].item()] = y[n, :]
