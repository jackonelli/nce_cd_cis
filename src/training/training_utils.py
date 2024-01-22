import os
from pathlib import Path
import torch
import numpy as np


# Metrices
class PrecisionErrorMetric:
    def __init__(self, true_precision):
        self.true_precison = true_precision

    def metric(self, model):
        return torch.mean((torch.exp(model.log_precision) - self.true_precison) ** 2)


class Mse:
    def __init__(self, true_model):
        self.mu = true_model.mu
        self.cov = true_model

    def metric(self, model):
        return torch.mean((self.mu - model.mu)**2)


class KlDiv:
    def __init__(self, true_model):
        self.true_model = true_model

    def metric(self, model):
        return torch.mean((torch.exp(model.log_precision) - self.true_precison) ** 2)
        

class PolynomialLr:
    """Polynomial decaying lr according to keras"""
    def __init__(self, decay_steps, initial_lr, end_lr=1e-7, power=1.0):
        self.decay_steps = decay_steps
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.power = power

    def decayed_learning_rate(self, step):
        step = min(step, self.decay_steps)
        return ((self.initial_lr - self.end_lr) * (1 - step / self.decay_steps)**self.power) + self.end_lr



# Stopping conditions
def no_stopping(new_params, old_params):
    return False


# From https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/cnce.m
def no_change_stopping_condition(new_params, old_params, tol=1e-4):

    step_condition = torch.sqrt(torch.sum((new_params - old_params)**2)) < torch.sqrt(torch.sum(new_params**2)) * tol
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
        assert prev.shape[0] == addition.shape[0], "Addition does not match previously saved array"
        np.save(file_name, np.column_stack((prev, addition)))
    else:
        # TODO: check that directory exists
        np.save(file_name, addition)


def remove_file(file_name):
    if os.path.exists(Path(file_name)):
        os.remove(file_name)


def normalised_log_likelihood(data_loader, criterion):

    model = criterion.get_model()

    #with torch.no_grad: # or put model in eval mode
    log_p_tilde = 0
    log_z = 0
    N = 0
    for i, (y, idx) in enumerate(data_loader, 0):
        log_p_tilde += model.log_prob(y).sum()
        log_z += criterion.outer_part_fn(y) * y.shape[0]

        N += y.shape[0]

    return log_p_tilde - log_z / N

