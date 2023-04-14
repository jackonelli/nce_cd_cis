import unittest
import torch

from src.experiments.ace_exp_utils import UniformMaskGenerator


class TestAceProposal(unittest.TestCase):
    def test_uniform_mask(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_samples = 10000

        mask = UniformMaskGenerator()(num_samples, num_features)

        mask_sum = torch.sum(mask, dim=-1)

        print(num_features)
        print(mask_sum.max())
        print(mask_sum.min())

        inds = torch.randperm(num_features)[:1]
        print(inds)


if __name__ == "__main__":
    unittest.main()