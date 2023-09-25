import torch
from torch import Tensor
from src.models.rbm.rbm import Rbm
from src.noise_distr.base import NoiseDistr


class RbmNoiseDistr(NoiseDistr):
    """RBM noise distr. """

    def __init__(self, rbm: Rbm):
        self._inner_distr = rbm

    def sample(self, size: torch.Size, x: Tensor):
        x = torch.repeat_interleave(x, size[-1], dim=0)
        return self._inner_distr.sample(x).detach().clone()

    def log_prob(self, samples, x: Tensor):
        # TODO: Eller kan jag på något sätt beräkna conditional prob? Men då måste jag spara h (om jag inte kan marginalisera över denna)
        return self._inner_distr.log_prob(samples).detach().clone()