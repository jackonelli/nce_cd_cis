import unittest
import torch

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.nce.cd_rank import CdRankCrit
from src.part_fn_utils import unnorm_weights, concat_samples

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestCdRank(unittest.TestCase):

    def test_order_grad_mean(self):
        """Test that gradient of mean is same as mean of gradient"""

        # Sample some data to test on
        num_samples = 5
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 20
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Get some neg. samples
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        noise_distr = MultivariateNormal(mu_noise, cov_noise)
        y_samples = noise_distr.sample(torch.Size((num_samples, num_neg_samples)), y)

        # Multivariate normal model
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)

        # Calculate weights
        ys = concat_samples(y, y_samples)

        w_tilde = unnorm_weights(ys, true_distr.prob, noise_distr.prob).detach()
        w = w_tilde / w_tilde.sum(dim=1, keepdim=True)

        res = true_distr.grad_log_prob(ys, w)

        # Reference calculation (mean of gradient)
        ref_mu = torch.zeros(mu_true.shape)
        ref_cov = torch.zeros(cov_true.shape)
        for i in range(num_samples):
            for j in range(num_neg_samples + 1):
                grads = true_distr.grad_log_prob(ys[i, j, :], w[i, j])
                ref = true_distr.grad_log_prob(ys[i, j, :])
                self.assertTrue(torch.allclose(grads[0], w[i, j] * ref[0]))
                self.assertTrue(torch.allclose(grads[1], w[i, j] * ref[1]))
                ref_mu += grads[0]
                ref_cov += grads[1]

        refs = [ref_mu / (num_samples * (num_neg_samples + 1)), ref_cov / (num_samples * (num_neg_samples + 1))]

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))

    def test_criterion_grad(self):
        """Check that criterion gives same gradient as NCE ranking for 1 step"""

        # Sample some data to test on
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true.clone(), cov_true.clone())
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        mcmc_steps = 1
        criterion = CdRankCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        y_samples = criterion.sample_noise((y.size(0), num_neg_samples), y)

        # Calculate gradient directly using CD+NCE ranking
        criterion.calculate_inner_crit_grad(y, y_samples, None)
        res = criterion.get_model_gradients()

        # Calculate gradient of NCE ranking crit.
        true_distr_ref = GaussianModel(mu_true.clone(), cov_true.clone())
        criterion_ref = NceRankCrit(true_distr_ref, noise_distr, num_neg_samples)
        criterion_ref.calculate_inner_crit_grad(y, y_samples, None)
        refs = criterion_ref.get_model_gradients()

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))

    def test_several_steps(self):
        """Just check that everything seems to run for multiple MCMC steps"""
        num_samples = 1000
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
        num_neg_samples = ((max_neg_samples - min_neg_samples) * torch.rand(1) + min_neg_samples).int()

        # Multivariate normal model and noise distr.
        mu_true, cov_true = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        mu_noise, cov_noise = torch.randn((y.shape[-1],)), torch.eye(y.shape[-1])
        true_distr = GaussianModel(mu_true, cov_true)
        noise_distr = MultivariateNormal(mu_noise, cov_noise)

        mcmc_steps = 3
        criterion = CdRankCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        criterion.calculate_crit_grad(y, None)


if __name__ == '__main__':
    unittest.main()
