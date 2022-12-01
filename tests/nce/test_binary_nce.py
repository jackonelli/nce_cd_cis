import unittest
import torch
import numpy as np

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.nce.binary import NceBinaryCrit
from src.part_fn_base import unnorm_weights


class TestBinaryNCE(unittest.TestCase):
    def test_criterion_equal_distr(self):
        """Check that criterion is correct for case model=noise distr."""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)
        
        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Set model and noise distr. to be equal
        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criterion = NceBinaryCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        res = criterion.crit(y)
        
        # For model and noise_distr equal, criterion should depend only on the number of neg. samples
        ref = - torch.log(1 / (1 + num_neg_samples)) - num_neg_samples * torch.log(num_neg_samples / (1 + num_neg_samples))       

        self.assertTrue(torch.allclose(ref, res))

    def test_criterion_example(self):
        """Test example for calculating NCE binary criterion"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)
        
        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criterion = NceBinaryCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise(num_neg_samples * y.size(0), y)
        res = criterion.inner_crit(y, y_samples)

        # Reference calculation (check so that positive and negative samples are used correctly)
        # Positive sample term
        y_w = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ref_y = torch.log(y_w / (y_w + num_neg_samples)).mean()

        # Negative sample term
        ys_w = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)
        ref_ys = num_neg_samples * torch.log(num_neg_samples / (ys_w + num_neg_samples)).mean()
        ref = - ref_y - ref_ys

        self.assertTrue(torch.allclose(ref, res))


def sample_postive_test_samples(num_samples, min_num_dims=2, max_num_dims=5):

    num_dims = np.random.randint(min_num_dims, max_num_dims)
    mu = torch.randn((num_dims,))
    y = torch.randn((num_samples, num_dims)) + mu

    return y



if __name__ == '__main__':
    unittest.main()
