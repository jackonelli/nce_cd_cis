import unittest
import torch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal


class TestCondMultivariateNormalDistr(unittest.TestCase):
    def test_dim(self):

        N, J, D = 2, 4, 3
        cov = torch.eye(D)
        distr = ConditionalMultivariateNormal(cov)
        y = torch.zeros((N, D))
        y_sample = distr.sample((N, J), y.reshape(y.size(0), 1, -1))
        self.assertEqual(y_sample.size(), (N, J, D))

    def test_cond_(self):
        N, J, D = 5, 10, 3
        y_sample = torch.ones((N, J, D))
        yp = torch.ones((N, 1, D))

        noise_distr = ConditionalMultivariateNormal(
            cov=3 * torch.eye(D),
        ).prob

        p_n = noise_distr(y_sample, yp)

    # def _log_unnorm_w(self, y, y_samples) -> Tensor:
    #     """Log weights of y (NxD) and y_samples (NxJxD)"""

    #     return log_cond_unnorm_weights(
    #         y.reshape(y.size(0), 1, -1),
    #         y_samples,
    #         self._unnorm_distr.log_prob,
    #         self._noise_distr.log_prob,
    #     )
