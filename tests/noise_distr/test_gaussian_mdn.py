import unittest
import torch
from src.noise_distr.gaussian_mdn import OneDFixedMdn


class TestOneDFixedMdn(unittest.TestCase):
    def test_init(self):
        # Test success
        parameters = [(0.3, -1.0, 2.0), (0.7, 1.0, 3.0)]
        mdn = OneDFixedMdn(parameters)

        self.assertTrue(torch.allclose(mdn.weights, torch.tensor([0.3, 0.7])))
        self.assertTrue(torch.allclose(mdn.means, torch.tensor([-1.0, 1.0])))
        self.assertTrue(torch.allclose(mdn.sigma_sqs, torch.tensor([2.0, 3.0])))

        # Check weights that sum to one but are out of (0.0, 1.0) range
        parameters = [(-0.3, -1.0, 2.0), (1.3, 1.0, 3.0)]
        with self.assertRaises(AssertionError):
            OneDFixedMdn(parameters)

        # Check weights that don't sum to 1.0
        parameters = [(0.3, -1.0, 2.0), (0.3, 1.0, 3.0)]
        with self.assertRaises(AssertionError):
            OneDFixedMdn(parameters)

        # Check negative variances
        parameters = [(0.3, -1.0, -2.0), (0.7, 1.0, 3.0)]
        with self.assertRaises(AssertionError):
            OneDFixedMdn(parameters)

    def test_sample(self):
        parameters = [(0.3, -1.0, 2.0), (0.7, 1.0, 3.0)]
        mdn = OneDFixedMdn(parameters)
        size = torch.Size([5, 10])
        y_samples = mdn.sample(size)
        self.assertEqual(size, y_samples.size())
