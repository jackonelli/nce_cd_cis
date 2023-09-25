from torch.utils.data import Dataset


class Generic(Dataset):
    """Generic dataset."""

    def __init__(self, data):
        """
        Args:
            data (Tensor): dataset with shape (N, D)
        """
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):

        sample = self.data[idx, :]

        return sample, idx
