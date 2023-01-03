import torch
import numpy as np


# Generation of true parameters, parameter initialisation
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/data_generation/generate_parameters.m
def generate_true_params(mu_min=5, mu_max=10, sigma_min=0.3, sigma_max=1.5):
    """Generate true parameters of ring model"""
    mu = (mu_max - mu_min) * torch.rand(1) + mu_min
    sigma = (sigma_max - sigma_min) * torch.rand(1) + sigma_min
    precision = sigma ** (-2)
    z = -0.5 * torch.log(2 * torch.tensor(np.pi)) - torch.log(sigma)

    return mu, precision, z


def initialise_params(mu_min=6, mu_max=8, sigma_min=0.3, sigma_max=1.5, z_min=0.01):
    """Initialise parameters of ring model"""
    mu = (mu_max - mu_min) * torch.rand(1) + mu_min
    sigma = (sigma_max - sigma_min) * torch.rand(1) + sigma_min
    precision = sigma ** (-2)

    z = torch.rand(1) + z_min

    return mu, torch.log(precision), torch.log(z)
