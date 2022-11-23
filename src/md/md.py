"""Mixed Distribution (MD)"""
import torch
from torch import Tensor
from src.part_fn_base import PartFnEstimator


class MdCrit(PartFnEstimator):
    def __init__(self, unnorm_distr, noise_distr, est_part):
        self._unnorm_distr = unnorm_distr
        self._noise_distr = noise_distr
        self._est_part = est_part

    def part_fn(self, y, y_samples) -> Tensor:
        y = y.reshape((1,))
        ys = torch.cat((y, y_samples))
        num_neg = y_samples.size(0)

        num = self._unnorm_distr(ys)
        noise_prob = self._noise_distr(ys)
        md_prob = self._unnorm_distr(ys) / self._est_part
        den = num_neg / (num_neg + 1) * noise_prob + 1 / (num_neg + 1) * md_prob

        return (num / den).sum() / (num_neg + 1)
