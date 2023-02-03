"""Energy-based model (EBM) interface

The model is specified as

    p_theta(y | x) = p_tilde(y | x) / Z_theta(x),

where the unnormalised model is parameterised by the energy E_theta(y, x):

    p_tilde_theta(y| x) = exp(- E_theta(y, x))
"""
from abc import abstractmethod
from typing import Optional
import torch
from torch import Tensor
from src.models.base_model import BaseModel


class Ebm(BaseModel):
    def __init__(self, pos_map=torch.exp):
        self._pos_map = pos_map

    def log_prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        """Compute p_tilde_theta(y | x)

        The unnormalised probability of a sample and is computed by
        mapping the negative energy to R^+
        """
        return -self.energy(y, x)

    @abstractmethod
    def energy(self, y: Tensor, x: Optional[Tensor]) -> Tensor:
        """Compute energy"""
        pass
