import unittest
import torch
import numpy as np

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.part_fn_base import norm_weights
from src.models.ebm.normal_params import NormalEbm


class TestPersistentCnce(unittest.TestCase):
    def test_y_persistent_update(self):
        J = 5
        crit = PersistentCondNceCrit(None, None, J)


if __name__ == "__main__":
    unittest.main()
