import unittest
import torch
import numpy as np

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.per_cnce import PersistentCondNceCrit
from src.part_fn_utils import norm_weights
from src.models.ebm.normal_params import NormalEbm


class TestPersistentCnce(unittest.TestCase):
    def test_criteria_equal_distr(self):
        pass


if __name__ == "__main__":
    unittest.main()
