import numpy as np
from torch.utils.data import Dataset


class UCIDataset(Dataset):
    """UCI datasets."""

    def  __init__(self, name="gas", set="train", root_dir='./data/datasets/uci/', transform=None, noise_scale=0.0):
        """
        Args:
            name: name of dataset (dsps, gas, hepmass, miniboone or power)
            set: which dataset (training, validation or test) to load ("train"/"val"/"test")
            root_dir: (string) directory for loading/saving data.
            transform: (callable, optional) optional data transform.
        """
        available_datasets = ["bsds", "gas", "hepmass", "miniboone", "power"]

        self.name = name
        self.set = set
        self.root_dir = root_dir
        self.transform = transform
        self.noise_scale = noise_scale

        if self.noise_scale > 0.0:
            self.transform = GaussianNoiseTransform(noise_scale).transform
        else:
            self.transform = None

        if self.name in available_datasets:

            if set not in ["train", "val", "test"]:
                print("Unknown set: loading train data.")
                self.set = "train"

            self.y = np.loadtxt(self.root_dir + self.name + "_" + self.set + ".txt")

            self.num_samples = self.y.shape[0]
            self.num_features = self.y.shape[-1]

        else:
            print("Unknown dataset, could not load data.")
            self.y = None
            self.num_samples = 0
            self.num_features = 0

    def get_full_data(self):
        return self.y

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        sample = self.y[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class GaussianNoiseTransform:
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale

    def transform(self, y):
        return y + np.random.normal(loc=0.0, scale=self.noise_scale, size=y.shape)

