"""Partition function estimator interface"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch import Tensor

from src.noise_distr.base import NoiseDistr
from src.models.base_model import BaseModel


class PartFnEstimator(ABC):
    def __init__(
        self, unnorm_distr: BaseModel, noise_distr: NoiseDistr, num_neg_samples: int
    ):

        self._unnorm_distr = unnorm_distr
        self._noise_distr = noise_distr
        self._num_neg = num_neg_samples

    def log_part_fn(self, y, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise(self._num_neg, y.reshape(y.size(0), 1, -1))

        return self.inner_log_part_fn(y, y_samples)

    def inner_log_part_fn(self, y, y_samples) -> Tensor:
        return torch.log(self.part_fn(y, y_samples))

    def crit(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise(self._num_neg, y.reshape(y.size(0), 1, -1))

        return self.inner_crit(y, y_samples)

    @abstractmethod
    def inner_crit(self, y: Tensor, y_samples: Tensor) -> Tensor:
        pass

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        self.crit(y, _idx).backward()

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        self.inner_crit(y, y_samples).backward()

    def get_model_gradients(self):
        return self._unnorm_distr.get_gradients()

    def outer_part_fn(self, y: Tensor, _idx: Optional[Tensor]) -> Tensor:
        y_samples = self.sample_noise(self._num_neg, y.reshape(y.size(0), 1, -1))

        return self.inner_crit(y, y_samples)


    @abstractmethod
    def part_fn(self, y, y_samples) -> Tensor:
        pass

    def sample_noise(self, num_samples: int, y: Tensor):
        # Note: num_samples = samples / obs.

        return self._noise_distr.sample(torch.Size((y.size(0), num_samples)), y.reshape(y.size(0), 1, -1))

    def get_model(self):
        return self._unnorm_distr


