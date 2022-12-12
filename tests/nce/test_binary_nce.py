import unittest
import torch
import numpy as np

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.models.ring_model.ring_model import RingModelNCE
from src.data.ring_model_dataset import RingModelDataset
from src.nce.binary import NceBinaryCrit
from src.part_fn_utils import unnorm_weights


class TestBinaryNCE(unittest.TestCase):
    def test_criterion_equal_distr(self):
        """Check that criterion is correct for case model=noise distr."""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Set model and noise distr. to be equal
        mu, cov = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu, cov)
        noise_distr = MultivariateNormal(mu, cov)
        criterion = NceBinaryCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        res = criterion.crit(y, None)

        # For model and noise_distr equal, criterion should depend only on the number of neg. samples
        ref = -torch.log(1 / (1 + num_neg_samples)) - num_neg_samples * torch.log(
            num_neg_samples / (1 + num_neg_samples)
        )

        self.assertTrue(torch.allclose(ref, res))

    def test_criterion_example(self):
        """Test example for calculating NCE binary criterion"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        criterion = NceBinaryCrit(true_distr, noise_distr, num_neg_samples)

        # Evaluate criterion
        y_samples = criterion.sample_noise(num_neg_samples, y)
        res = criterion.inner_crit(y, y_samples)

        # Reference calculation (check so that positive and negative samples are used correctly)
        # Positive sample term
        y_w = unnorm_weights(y, true_distr.prob, noise_distr.prob)
        ref_y = torch.log(y_w / (y_w + num_neg_samples)).mean()

        # Negative sample term
        ys_w = unnorm_weights(y_samples, true_distr.prob, noise_distr.prob)
        ref_ys = (
            num_neg_samples
            * torch.log(num_neg_samples / (ys_w + num_neg_samples)).mean()
        )
        ref = -ref_y - ref_ys

        self.assertTrue(torch.allclose(ref, res))

    def test_grad_crit(self):
        """Test so that gradient is calculated correctly during training"""

        num_samples = 10
        num_dims = 2  # np.random.randint(2, 5)

        # Get model
        mu, log_precision = torch.randn(1), torch.randn(1) + 1e-3

        # Get data
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = (
            (max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples
        ).int()

        training_data = RingModelDataset(
            sample_size=num_samples,
            num_dims=num_dims,
            mu=mu.numpy(),
            precision=torch.exp(log_precision).numpy(),
            root_dir="test_data",
        )
        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=2, shuffle=False
        )

        # Initialise model
        log_precision_init, log_z_init = torch.randn(1) + 1e-3, torch.randn(1) + 1e-3
        model = RingModelNCE(mu, log_precision_init, log_z_init)

        # Get noise distr. params
        mu_noise, cov_noise = torch.randn((num_dims,)), torch.eye(num_dims)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        # Get criterion
        criterion = NceBinaryCrit(model, noise_distr, num_neg_samples)

        # Run training for one epoch and check final parameters
        optimizer = torch.optim.SGD(criterion.get_model().parameters(), lr=0.1)
        for i, (y, idx) in enumerate(train_loader, 0):

            y_samples = criterion.sample_noise(num_neg_samples, y)

            # Calculate using gradient function
            optimizer.zero_grad()
            criterion.calculate_inner_crit_grad(y, y_samples)
            res = [grad.detach().clone() for grad in criterion.get_model_gradients()]

            # Calculate "as usual"
            optimizer.zero_grad()
            loss = criterion.inner_crit(y, y_samples)
            loss.backward()
            refs = [grad.detach().clone() for grad in criterion.get_model_gradients()]

            optimizer.step()

            for param, param_ref in zip(res, refs):
                self.assertTrue(torch.allclose(param_ref, param, rtol=1e-4))


def sample_postive_test_samples(num_samples, min_num_dims=2, max_num_dims=5):

    num_dims = np.random.randint(min_num_dims, max_num_dims)
    mu = torch.randn((num_dims,))
    y = torch.randn((num_samples, num_dims)) + mu

    return y


if __name__ == "__main__":
    unittest.main()
