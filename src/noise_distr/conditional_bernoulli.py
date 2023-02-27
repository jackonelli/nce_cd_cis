import torch
from torch import Tensor
from src.noise_distr.base import NoiseDistr


class ConditionalMultivariateBernoulli(NoiseDistr):
    def __init__(self, p_0: Tensor, p_1: Tensor):
        """Distribution with parameters p_0 if  conditioned on x=0, distribution with parameters p_1 otherwise"""

        self.p_0 = p_0
        self.p_1 = p_1
        self.dim = p_0.shape[0]

        self._inner_distr = torch.distributions.Bernoulli(p_0)
        self._inner_distr_1 = torch.distributions.Bernoulli(p_1)

    def sample(self, size: torch.Size, x: Tensor):
        y_sample = (1 - x) * self._inner_distr.sample(size) + x * self._inner_distr_1.sample(size)
        return y_sample

    def log_prob(self, samples, x: Tensor):

        assert samples.size(-1) == self.dim
        return ((1 - x) * self._inner_distr.log_prob(samples) + x * self._inner_distr_1.log_prob(samples)).sum(dim=-1)


class ClassConditionalMultivariateBernoulli(NoiseDistr):
    def __init__(self, p_0: Tensor, p_1: Tensor):
        """Distribution with parameters p_0 if  conditioned on x=0, distribution with parameters p_1 otherwise"""

        self.p_0 = p_0
        self.p_1 = p_1
        self.dim = p_0.shape[-1]

    def filtered_probs(self, x: Tensor):
        return torch.matmul(x, self.p_0).squeeze(dim=1), torch.matmul(x, self.p_1).squeeze(dim=1)

    def sample(self, size: torch.Size, x: tuple):
        y, x = x

        p_0, p_1 = self.filtered_probs(x)
        p_0, p_1 = p_0.unsqueeze(dim=1).repeat(1, size[1], 1), p_1.unsqueeze(dim=1).repeat(1, size[1], 1)
        inner_distr = torch.distributions.Bernoulli(p_0)
        inner_distr_1 = torch.distributions.Bernoulli(p_1)

        y_sample = (1 - y) * inner_distr.sample() + y * inner_distr_1.sample()

        assert y_sample.shape == (size[0], size[1], y.shape[-1])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        import torchvision
        import numpy as np

        ax[0].imshow(
            np.transpose(torchvision.utils.make_grid(y[:8, :].reshape(-1, 1, 28, 28), nrow=4).numpy(), (1, 2, 0)))
        ax[1].imshow(
            np.transpose(torchvision.utils.make_grid(y_sample[:8, :].reshape(-1, 1, 28, 28), nrow=4).numpy(), (1, 2, 0)))

        plt.show()

        return y_sample

    def log_prob(self, samples, x: tuple):
        y, x = x

        p_0, p_1 = self.filtered_probs(x)
        p_0, p_1 = p_0.unsqueeze(dim=1).repeat(1, samples.shape[1], 1), p_1.unsqueeze(dim=1).repeat(1, samples.shape[1],
                                                                                                    1)
        inner_distr = torch.distributions.Bernoulli(p_0)
        inner_distr_1 = torch.distributions.Bernoulli(p_1)

        assert samples.size(-1) == self.dim
        return ((1 - y) * inner_distr.log_prob(samples) + y * inner_distr_1.log_prob(samples)).sum(dim=-1)


class SpatialConditionalMultivariateBernoulli(NoiseDistr):
    def __init__(self, p_0: Tensor, p_1: Tensor):
        """Distribution with parameters p_0 if  conditioned on x=0, distribution with parameters p_1 otherwise"""

        self.p_0 = p_0
        self.p_1 = p_1
        self.dim = p_0.shape[0]

    def filtered_probs(self, x: Tensor):
        return torch.matmul(x, self.p_0).squeeze(dim=1), torch.matmul(x, self.p_1).squeeze(dim=1)

    def sample(self, size: torch.Size, x: tuple):
        y, x = x

        p_0, p_1 = self.filtered_probs(x)
        inner_distr = torch.distributions.Bernoulli(p_0)
        inner_distr_1 = torch.distributions.Bernoulli(p_1)

        dim = torch.sqrt(torch.tensor(x.size(-1))).int()
        y = mean_filter(x.reshape(-1, 1, dim, dim)).reshape(x.size())
        y[y >= 0.5] = 1.0 # TODO: Vi kan väl kalla detta majority vote?
        y[y < 0.5] = 0.0
        y_sample = (1 - y) * inner_distr.sample(size) + y * inner_distr_1.sample(size)

        return y_sample

    def log_prob(self, samples, x: tuple):
        y, x = x

        p_0, p_1 = self.filtered_probs(x)
        p_0, p_1 = p_0.unsqueeze(dim=1).repeat(1, samples.shape[1], 1), p_1.unsqueeze(dim=1).repeat(1, samples.shape[1],
                                                                                                    1)
        inner_distr = torch.distributions.Bernoulli(p_0)
        inner_distr_1 = torch.distributions.Bernoulli(p_1)

        dim = torch.sqrt(torch.tensor(x.size(-1))).int()
        y = mean_filter(x.reshape(-1, 1, dim, dim)).reshape(x.size())

        y[y >= 0.5] = 1.0
        y[y < 0.5] = 0.0

        assert samples.size(-1) == self.dim
        return ((1 - y) * inner_distr.log_prob(samples) + y * inner_distr_1.log_prob(samples)).sum(dim=-1)


class SpatialClassConditionalMultivariateBernoulli(NoiseDistr):
    def __init__(self, p_0: Tensor, p_1: Tensor):
        """Distribution with parameters p_0 if  conditioned on x=0, distribution with parameters p_1 otherwise"""

        self.p_0 = p_0
        self.p_1 = p_1
        self.dim = p_0.shape[0]

        self._inner_distr = torch.distributions.Bernoulli(p_0)
        self._inner_distr_1 = torch.distributions.Bernoulli(p_1)

    def sample(self, size: torch.Size, x: Tensor):
        dim = torch.sqrt(torch.tensor(x.size(-1))).int()
        y = mean_filter(x.reshape(-1, 1, dim, dim)).reshape(x.size())
        y[y >= 0.5] = 1.0 # TODO: Vi kan väl kalla detta majority vote?
        y[y < 0.5] = 0.0
        y_sample = (1 - y) * self._inner_distr.sample(size) + y * self._inner_distr_1.sample(size)
        return y_sample

    def log_prob(self, samples, x: Tensor):
        dim = torch.sqrt(torch.tensor(x.size(-1))).int()
        y = mean_filter(x.reshape(-1, 1, dim, dim)).reshape(x.size())

        y[y >= 0.5] = 1.0
        y[y < 0.5] = 0.0

        assert samples.size(-1) == self.dim
        return ((1 - y) * self._inner_distr.log_prob(samples) + y * self._inner_distr_1.log_prob(samples)).sum(dim=-1)


def mean_filter(x):
  """
  Calculating the mean of each 3x3 neighborhood.
  input:
    - x_bchw: input tensor of dimensions batch-channel-height-width
  output:
    - y_bchw: each element in y is the average of the 9 corresponding elements in x_bchw
  """
  # define the filter
  filter_size = 5
  box = torch.ones((filter_size, filter_size), dtype=x.dtype, device=x.device, requires_grad=False)
  box = box / box.sum()
  box = box[None, None, ...].repeat(x.size(1), 1, 1, 1)
  # use grouped convolution - so each channel is averaged separately.
  padding = int((filter_size - 1) / 2)
  y = torch.nn.functional.conv2d(x, box, padding=padding, groups=x.size(1))
  return y