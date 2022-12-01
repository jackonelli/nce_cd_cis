import unittest
import torch
import numpy as np

from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit
from src.part_fn_base import norm_weights


class TestPersistentCnce(unittest.TestCase):
    def test_criteria_equal_distr(self):
        pass


def y_weights_norm(y, y_samples, true_distr, noise_distr):
    y_w_tilde = unnorm_weights(y, true_distr.prob, noise_distr.prob)
    return y_w_tilde / (
        y_w_tilde + unnorm_weights(y_samples, true_distr.prob, noise_distr.prob).sum()
    )


if __name__ == "__main__":
    unittest.main()
