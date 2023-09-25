# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.data.data_uci.uciutils import preprocess_and_save_miniboone, get_data_root

class MiniBooNEDataset(Dataset):
    def __init__(self, split='train', frac=None, num_dims=None, scaling='standardize'):
    
        if scaling == 'standardize':
            path = os.path.join(get_data_root(), 'data', 'miniboone/{}.npy'.format(split))
        elif scaling == 'normalize':
            path = os.path.join(get_data_root(), 'data_norm', 'miniboone/{}.npy'.format(split))

        try:
            self.data = np.load(path).astype(np.float32)
            print("Data loaded from:" + str(path))
        except FileNotFoundError:
            print('Preprocessing and saving Miniboone...')
            preprocess_and_save_miniboone(scaling)
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
    dataset = MiniBooNEDataset(split='train')
    print(type(dataset.data))
    print(dataset.data.shape)
    print(dataset.data.min(), dataset.data.max())
    plt.hist(dataset.data.reshape(-1), bins=250)
    plt.show()


def main():
    test()


if __name__ == '__main__':
    main()
