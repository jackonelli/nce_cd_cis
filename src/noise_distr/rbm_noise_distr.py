import torch
from torch import Tensor
from src.models.rbm.rbm import Rbm
from src.noise_distr.base import NoiseDistr


class RbmNoiseDistr(NoiseDistr):
    """RBM noise distr. """

    def __init__(self, rbm: Rbm):
        self._inner_distr = rbm

    def sample(self, size: torch.Size, x: Tensor):
        h = torch.rand((size[0] * size[-1], 1, self._inner_distr.weights.shape[-1]))
        return self._inner_distr.sample_from_hidden(h).detach().clone()

    def log_prob(self, samples, x: Tensor):
        return self._inner_distr.log_prob(samples).detach().clone()


class ConditionalRbmNoiseDistr(NoiseDistr):
    """RBM noise distr. """

    def __init__(self, rbm: Rbm):
        self._inner_distr = rbm

    def sample(self, size: torch.Size, x: Tensor):
        x = torch.repeat_interleave(x, size[-1], dim=0)
        _, y_sample = self._inner_distr.sample(x).detach().clone()
        return y_sample

    def log_prob(self, samples, x: Tensor):
        # TODO: Kan jag på något sätt beräkna conditional prob? Men då måste jag spara h (om jag inte kan marginalisera över denna)
        p_h = self._inner_distr.p_h.detach().clone()
        h = self._inner_distr.h.detach().clone()
        log_p_h = (h * torch.log(p_h) + (1 - h) * torch.log(1 - p_h)).sum(-1)

        p_v, _ = self._inner_distr.sample_visible(h)
        log_p_x = (x * torch.log(p_v) + (1 - x) * torch.log(1 - p_v)).sum(-1)

        return log_p_x * log_p_h