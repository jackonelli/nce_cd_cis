import unittest
import torch

from src.models.gaussian_model import GaussianModel
from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.nce.cd_rank import CdRankCrit
from src.part_fn_utils import unnorm_weights, concat_samples

from tests.nce.test_binary_nce import sample_postive_test_samples


class TestCdRank(unittest.TestCase):

    # TODO: Why isn't this the same?
    def test_order_grad_mean(self):
        """Test that gradient of mean is same as mean of gradient"""

        # Sample some data to test on
        num_samples = 10
        y = sample_postive_test_samples(num_samples)

        # Random number of negative samples
        min_neg_samples, max_neg_samples = 2, 5
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

        # TODO: Varför verkar bara ett samples inkluderas när jag tar väntevärde?
        print(true_distr.log_prob(ys[0, :2]).mean())
        print(torch.tensor([true_distr.log_prob(ys[0, 0]), true_distr.log_prob(ys[0, 1])]).mean())

        w_tilde = unnorm_weights(ys, true_distr.prob, noise_distr.prob).detach()
        w = w_tilde / w_tilde.sum(dim=1, keepdim=True)

        # Calculate gradients of log prob (gradient of mean)
        # TODO: DET ÄR SOM ATT GRADIENTEN BARA BERÄKNAS FÖR FÖRSTA SAMPLET???
        res = true_distr.grad_log_prob(ys, w)
        res_2 = true_distr.grad_log_prob(ys[0, 0], w[0, 0]) #true_distr.grad_log_prob(ys, w)

        # Just check that everything stays constant as expected
        for grad, grad_2 in zip(res, res_2):
            self.assertTrue(torch.allclose(grad, grad_2))

        # Reference calculation (mean of gradient)
        ref_mu = torch.zeros(mu_true.shape)
        ref_cov = torch.zeros(cov_true.shape)

        for i in range(1):
            for j in range(1):
                grads = true_distr.grad_log_prob(w[i, j] * ys[i, j, :])
                # TODO: BLIR INTE SAMMA OM JAG FLYTTAR UT VIKTEN??? TAS GRADIENT M.A.P VIKT?
                ref_mu += grads[0]
                ref_cov += grads[1]

        refs = [ref_mu, ref_cov]

        for grad, grad_ref in zip(res, refs):
            print(grad)
            print(grad_ref)
            #self.assertTrue(torch.allclose(grad_ref, grad))

    def test_criterion_grad(self):
        """Check that criterion gives same gradient as NCE ranking for 1 step"""

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

        mcmc_steps = 1
        criterion = CdRankCrit(true_distr, noise_distr, num_neg_samples, mcmc_steps)
        y_samples = criterion.sample_noise((y.size(0), num_neg_samples), y)

        # Calculate gradient directly using CD+NCE ranking
        criterion.calculate_inner_crit_grad(y, y_samples)
        res = criterion.get_model_gradients()

        # Calculate gradient of NCE ranking crit.
        true_distr_ref = GaussianModel(mu_true, cov_true)
        criterion_ref = NceRankCrit(true_distr_ref, noise_distr, num_neg_samples)
        criterion_ref.calculate_inner_crit_grad(y, y_samples)
        refs = criterion_ref.get_model_gradients()

        for grad, grad_ref in zip(res, refs):
            self.assertTrue(torch.allclose(grad_ref, grad, rtol=1e-4))


if __name__ == '__main__':
    unittest.main()
