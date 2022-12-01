import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class RingModelDataset(Dataset):
    """Ring model dataset."""

    def __init__(self, sample_size, num_dims, mu, precision, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.num_samples = sample_size
        self.y = generate_ring_data(num_samples=self.num_samples, num_dims=num_dims, mu=mu, precision=precision)
        np.save(self.root_dir, self.y)

    def get_full_data(self):
        return torch.tensor(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample = self.y[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample


def generate_ring_data(num_samples, num_dims, mu, precision):
    # Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/data_generation/genRingData.m
    # They use a cutoff of 0.1 for the radii

    u = np.random.normal(size=(num_samples, num_dims))
    u = u / np.sqrt(np.sum(u**2, axis=-1, keepdims=True))

    r = np.sqrt(1 / precision) * np.random.normal(size=(num_samples, 1)) + mu

    return u * r


def plot_2d_ring_data():
    dataset = RingModelDataset(sample_size=1000, num_dims=2, mu=3, precision=2, root_dir="datasets/ring_model_test")
    plt.plot(dataset.y[:, 0], dataset.y[:, 1], '.')
    plt.show()


if __name__ == '__main__':
    plot_2d_ring_data()