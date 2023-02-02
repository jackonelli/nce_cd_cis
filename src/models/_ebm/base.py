"""Energy-based model (EBM) interface"""
from abc import abstractmethod
import torch
from torch import Tensor
from src.models.unnormalised import UnnormModel


class Ebm(UnnormModel):
    def __init__(self, pos_map):
        self._pos_map = pos_map

    def unnorm_prob(self, y: Tensor) -> Tensor:
        """Compute p_tilde(y)

        The unnormalised probability of a sample and is computed by
        mapping the negative energy to R^+
        """
        return self._pos_map(-self.energy(y))

    @abstractmethod
    def energy(self, y: Tensor) -> Tensor:
        """Compute energy"""
        pass

    @abstractmethod
    def prob(self, y: Tensor) -> Tensor:
        pass

    def log_prob(self, y: Tensor) -> Tensor:
        return torch.log(self.prob(y))
