import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class RingModelDataset(Dataset):
    """Ring model dataset."""

    def __init__(self, sample_size, num_dims, mu, precision, data_path, transform=None):
        """
        Args:
            sample_size: (int) number of samples to generate
            num_dims: (int) dimension of data
            mu: (tensor float) model mean.
            precision: (tensor float) model precision
            data_path: (string) file for saving data
            transform: (callable, optional) optional data transform
        """
        self.data_path = data_path
        self.data_path.parent.mkdir(exist_ok=True, parents=True)
        self.transform = transform

        self.num_samples = sample_size
        self.y = generate_ring_data(
            num_samples=self.num_samples, num_dims=num_dims, mu=mu, precision=precision
        )
        np.save(self.data_path, self.y.numpy())

    def get_full_data(self):
        return self.y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample = self.y[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


def generate_ring_data(num_samples, num_dims, mu, precision):
    # Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/data_generation/genRingData.m
    # They use a cutoff of 0.1 for the radii

    u = torch.randn(size=(num_samples, num_dims))
    u = u / torch.sqrt(torch.sum(u**2, dim=-1, keepdim=True))

    r = torch.sqrt(1 / precision) * torch.randn(size=(num_samples, 1)) + mu

    return u * r


def plot_2d_ring_data():
    dataset = RingModelDataset(
        sample_size=1000,
        num_dims=2,
        mu=3,
        precision=2,
        data_path="datasets/ring_model_test",
    )
    plt.plot(dataset.y[:, 0], dataset.y[:, 1], ".")
    plt.show()


if __name__ == "__main__":
    plot_2d_ring_data()
