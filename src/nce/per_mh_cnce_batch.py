"""Persistent Conditional Noise Contrastive Estimation (CNCE)

Inspired by Contrastive Divergence (CD) a persistent y is saved from the previous iteration.

The method works like CNCE (`CondNceCrit`) but it has a persistent sample yp
that is used to condition on when sampling noisy samples.
"""
from typing import Optional
import torch
from torch import Tensor
from torch.distributions import Categorical

from src.nce.cd_mh_cnce import CdMHCnceCrit
from src.utils.part_fn_utils import concat_samples, log_cond_unnorm_weights


class PersistentMHCnceCritBatch(CdMHCnceCrit):
    """Persistent cond. NCE crit"""

    def __init__(self, unnorm_distr, noise_distr, num_neg_samples: int, save_acc_prob=False, save_dir=None):
        mcmc_steps = 1  #TODO: If we want to take several MCMC-steps, persistent y should  be updated at end of gradient calculation
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, mcmc_steps, save_acc_prob, save_dir)
        self._persistent_y = dict()
        self.name = "pers_mh_cnce"

    def calculate_crit_grad(self, y: Tensor, idx: Optional[Tensor]) -> Tensor:
        assert (
            idx is not None
        ), "PersistentCondNceCrit requires an idx tensor that is not None"

        # TODO: The best would be to restructure y as a N*J x D matrix throughout this whole process (as in cd_cnce)
        #   This is in line with having pairs (y_0, y_1). Maybe, we could then make a dict with keys idx_0, ..., idx_J
        #   or similar

        # TODO: this is a bit of an override
        idx = torch.arange(start=0, end=y.shape[0])

        y = y.unsqueeze(dim=1).repeat(1, self._num_neg, 1)

        assert torch.allclose(y[0, 0, :], y[0, 1, :])

        y_p = self.persistent_y(y, idx).reshape(-1, y.shape[-1])
        y_samples = self.sample_noise(1, y_p)

        # NB We recompute w_tilde in inner_crit to comply with the API.
        w_y = torch.exp(-self._log_unnorm_w_ratio(y_p, y_samples).detach())
        w_y[w_y >= 1.0] = 1.0
        w = torch.cat((1 - w_y, w_y), dim=1).unsqueeze(dim=1)

        self._update_persistent_y(w, y_p, y_samples, idx)

        return self.calculate_inner_crit_grad(y_p, y_samples, y.reshape(-1, y.shape[-1]))

    def persistent_y(self, actual_y: Tensor, idx: Tensor):
        """Get persistent y

        Access the last selected y or return the y sampled from p_d
        if no last selected y exists
        """

        per_y = torch.empty(actual_y.size())
        for n, per_n in enumerate(idx):
            per_n = per_n.item()
            per_y[n, :, :] = (
                self._persistent_y[per_n]
                if self._persistent_y.get(per_n) is not None
                else actual_y[n, :, :]
            )
        return per_y

    def _update_persistent_y(self, w, y, y_samples, idx):
        """Sample new persistent y"""
        ys = concat_samples(y, y_samples)
        assert ys.shape == (y.shape[0], 2, y.shape[1])
        assert len(idx) == (y.shape[0] / self._num_neg)

        with torch.no_grad():
            for n, i in zip(range(len(idx)), range(0, y.shape[0], self._num_neg)):
                self._persistent_y[idx[n].item()] = torch.stack([ys[i+j,
                                                                 Categorical(probs=w[i+j, 0, :]).sample(), :]
                                                                 for j in range(self._num_neg)], dim=0)

