import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


class MnistDataset(Dataset):
    """MNIST dataset."""

    def __init__(self, train=True, root_dir='./data', transform=None):
        """
        Args:
            train (bool): if loading training (or test) data.
            root_dir (string): directory for saving data.
            transform (callable, optional): optional data transform.
        """
        self.root_dir = root_dir
        self.transform = transform

        set = torchvision.datasets.MNIST(root=root_dir, train=train, download=True)
        self.y = set.data.reshape(-1, 28**2) / 255
        self.num_samples = self.y.shape[0]

    def get_full_data(self):
        return torch.tensor(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample = self.y[idx, :]

        if self.transform:
            sample = self.transform(sample)

        img = torch.distributions.bernoulli.Bernoulli(sample).sample()

        return img, idx


def plot_mnist_example():
    """Check so that everything loads correctly"""

    dataset = MnistDataset(
        root_dir="datasets",
    )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=0)
    dataiter = iter(dataloader)
    images, _ = next(dataiter)

    # show images
    plt.imshow(np.transpose(torchvision.utils.make_grid(images.reshape(-1, 1, 28, 28)).numpy(), (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    plot_mnist_example()
