# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.data.data_uci.uciutils import preprocess_and_save_gas, get_data_root


class GasDataset(Dataset):
    def __init__(self, split='train', frac=None, num_dims=None, scaling = 'standardize'):
        
        if scaling == 'standardize':
            path = os.path.join(get_data_root(), 'data', 'gas/{}.npy'.format(split))
        elif scaling == 'normalize':
            path = os.path.join(get_data_root(), 'data_norm', 'gas/{}.npy'.format(split))
        
        try:
            self.data = np.load(path).astype(np.float32)
            print("Data loaded from:" + str(path))
        except FileNotFoundError:
            print('Preprocessing and saving Gas...')
            preprocess_and_save_gas(scaling)
            print('Done!')
            self.data = np.load(path).astype(np.float32)
            print("Data loaded from:" + str(path))

        if num_dims is not None:
            assert num_dims <= self.data.shape[-1], "Dimension is larger than number of features"
            self.data = self.data[:, :num_dims]

        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n


def test():
    dataset = GasDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    print(np.where(dataset.data == dataset.data.max()))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    axs = axs.reshape(-1)
    for i, dimension in enumerate(dataset.data.T):
        print(i)
        axs[i].hist(dimension, bins=100)
    plt.tight_layout()
    plt.show()


def main():
    test()


if __name__ == '__main__':
    main()