import unittest

import torch
from scipy.stats import truncnorm
from src.noise_distr.truncated_normal import TruncatedNormal

from matplotlib import pyplot as plt

class TestTruncNorm(unittest.TestCase):
    def test_sampling(self):

        dims = 1
        loc = torch.tensor([[-5.0641]])#-100.0 + torch.rand((dims, 1)) * 200.0
        scale = torch.tensor([[0.3413]])

        #scale = 0.1 + torch.rand((dims, 1)) * 5.0

        a = torch.tensor([[6.3288]]) * scale + loc #- 10.0 + torch.rand((dims, 1)) * 5.0
        b = torch.tensor([[67.7193]]) * scale + loc #5.0 + torch.rand((dims, 1)) * 5.0

        print(a)
        print(b)

        num_samples = 10000
        samples_1 = TruncatedNormal(loc=loc, scale=scale, a=a, b=b).sample((num_samples,))
        print(samples_1.shape)
        print(samples_1.min(dim=0))
        print(samples_1.max(dim=0))

        a_scaled, b_scaled = (a - loc) / scale, (b - loc) / scale
        samples_2 = torch.tensor(truncnorm.rvs(a_scaled.squeeze(), b_scaled.squeeze(), size=(num_samples, dims))) * scale.squeeze() + loc.squeeze()
        print(samples_2.shape)
        print(samples_2.min(dim=0))
        print(samples_2.max(dim=0))

        fig, ax = plt.subplots(2, dims)
        for i in range(dims):
            ax[0].hist(samples_1[:, i, 0])
            ax[1].hist(samples_2[:, i])
        plt.show()

if __name__ == "__main__":
    unittest.main()

