import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.noise_distr.normal import MultivariateNormal


class MultivariateNormalData(Dataset):
    """Virtual dataset that samples from a MVN distribution"""

    def __init__(self, mu: Tensor, cov: Tensor, num_samples: int):
        """
        Args:
            mu: (D,)
            cov: (D, D)
            num_samples: virtual size of the dataset, only used to create epochs with finite batch sizes.
        """
        self._distr = MultivariateNormal(mu, cov)
        self._num_samples = num_samples

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        # Note the returned idx has no meaning here, since we always resample.
        sample = self._distr.sample(torch.Size((1,)))
        return sample, idx
