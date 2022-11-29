import unittest
import torch
import numpy as np

from src.models.ring_model.ring_model import RingModel


class TestRingModel(unittest.TestCase):
    def test_construct_with_int(self):
        mu = 7
        precision = 1
        log_prec = np.log(precision)

        # p_m = RingModel(mu=mu, log_precision=log_prec)
