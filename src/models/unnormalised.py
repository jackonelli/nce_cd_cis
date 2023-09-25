"""Unnormalised model interface"""
from abc import ABC, abstractmethod
from torch import Tensor


class UnnormModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def unnorm_prob(self, y: Tensor) -> Tensor:
        """Compute unnorm prob p_tilde(y)"""
        pass
