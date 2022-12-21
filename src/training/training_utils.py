import os
from pathlib import Path
import torch
import numpy as np


# Metrices
class PrecisionErrorMetric:
    def __init__(self, true_precision):
        self.true_precison = true_precision

    def metric(self, model):
        return torch.mean(
            (torch.exp(model.log_precision) - self.true_precison) ** 2
        ).item()


class Mse:
    def __init__(self, true_mu):
        self.mu = true_mu

    def metric(self, model):
        return torch.mean((self.mu - model.mu) ** 2).item()


class MvnKlDiv:
    """KL divergence between two MVN distributions"""

    def __init__(self, true_mu, true_cov):
        self.dim = true_cov.size(0)
        self.true_mu = true_mu.reshape((self.dim, 1))
        self.true_cov = true_cov

    def metric(self, model):
        mu, cov = model.mu.reshape((self.dim, 1)), model.cov()
        term_1 = torch.logdet(cov) - torch.logdet(self.true_cov)
        term_2 = torch.trace(torch.linalg.solve(cov, self.true_cov))

        mu_diff = mu - self.true_mu
        cov_inv = torch.linalg.inv(cov)
        term_3 = mu_diff.T @ cov_inv @ mu_diff
        kl_div = (term_1 + term_2 + term_3 - self.dim) / 2
        return kl_div.item()


# Stopping conditions
def no_stopping(_new_params, _old_params):
    return False


# From https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/cnce.m
def no_change_stopping_condition(new_params, old_params, tol=1e-4):

    step_condition = (
        torch.sqrt(torch.sum((new_params - old_params) ** 2))
        < torch.sqrt(torch.sum(new_params ** 2)) * tol
    )
    if step_condition:
        return True
    else:
        return False


# This is for logging of results
def add_to_npy_file(file_name, addition):
    # Note: file_name should have ".npy" ending

    if addition.ndim == 1:
        addition.reshape(-1, 1)

    if os.path.exists(Path(file_name)):
        prev = np.load(file_name)
        assert (
            prev.shape[0] == addition.shape[0]
        ), "Addition does not match previously saved array"
        np.save(file_name, np.column_stack((prev, addition)))
    else:
        # TODO: check that directory exists
        np.save(file_name, addition)


def remove_file(file_name):
    if os.path.exists(Path(file_name)):
        os.remove(file_name)
