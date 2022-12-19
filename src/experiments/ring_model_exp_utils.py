import torch
import numpy as np

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit


# Noise distribution parameters, NCE
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gNCEnoise.m
def get_nce_noise_distr_par(y, eps=1e-10):
    mu = torch.mean(y, dim=0)
    cov = torch.cov(torch.transpose(y, 0, 1))

    if not (cov == cov.T).all():
        cov = (cov + cov.T) / 2

    #    if not (torch.eig(cov_noise_nce)[0][:,0] >= 0).all():
    cov = cov + eps * torch.eye(cov.shape[0])

    return mu, cov


# Noise distribution parameters, CNCE
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gNoise.m
def get_cnce_epsilon_base(y):
    return torch.std(y, dim=-1).mean()


def get_cnce_covariance_matrix(epsilon_factor, epsilon_base, num_dims):
    return torch.eye(num_dims) * (epsilon_factor * epsilon_base) ** 2


# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gEpsilonBaseFun.m
def get_cnce_noise_distr_par(y, neg_sample_size, mu, log_precision, model_class):
    epsilon_factor = get_cnce_epsilon_factor(y, neg_sample_size, mu, log_precision, model_class)
    epsilon_base = get_cnce_epsilon_base(y)
    return get_cnce_covariance_matrix(epsilon_factor, epsilon_base, y.size(-1))


def evaluate_cnce_loss(y, epsilon_factor, neg_sample_size, p_m):
    epsilon_base = get_cnce_epsilon_base(y)
    cov_noise = get_cnce_covariance_matrix(epsilon_factor, epsilon_base, y.size(-1))

    p_n = ConditionalMultivariateNormal(cov=cov_noise)
    criterion = CondNceCrit(p_m, p_n, neg_sample_size)

    return criterion.crit(y, 0)


def get_cnce_epsilon_factor(y, neg_sample_size, mu, log_precision, model_class, thrs_lower=0.05, thrs_upper=0.5,
                            inc_rate=0.2, dec_rate=0.5, max_iter=500, eps_hard_cap=1000):

    loss_zero = np.log(2)  # Loss as epsilon -> 0
    loss_inf = 0  # Loss as epsilon -> inf
    thrs_upper = thrs_upper * loss_zero

    p_m = model_class(mu=mu, log_precision=log_precision)

    epsilon_factor = 0.5  # Start value

    # Calculate initial loss
    loss = evaluate_cnce_loss(y, epsilon_factor, neg_sample_size, p_m)

    # Iterate until conditions are met
    k = 1
    while (k < max_iter) and (abs(1 - (loss / loss_zero)) < thrs_lower or loss < thrs_upper) and (
            epsilon_factor < eps_hard_cap):

        if abs(1 - (loss / loss_zero)) < thrs_lower:
            epsilon_factor = (1 + inc_rate) * epsilon_factor
        elif loss < thrs_upper:
            epsilon_factor = (1 - dec_rate) * epsilon_factor

        loss = evaluate_cnce_loss(y, epsilon_factor, neg_sample_size, p_m)

        k = k + 1

    return epsilon_factor


# Generation of true parameters, parameter initialisation
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/data_generation/generate_parameters.m
def generate_true_params(mu_min=5, mu_max=10, sigma_min=0.3, sigma_max=1.5):
    mu = (mu_max - mu_min) * torch.rand(1) + mu_min
    sigma = (sigma_max - sigma_min) * torch.rand(1) + sigma_min
    precision = sigma ** (-2)
    z = -0.5 * torch.log(2 * torch.tensor(np.pi)) - torch.log(sigma)

    return mu, precision, z


def initialise_params(mu_min=6, mu_max=8, sigma_min=0.3, sigma_max=1.5, z_min=0.01):
    mu = (mu_max - mu_min) * torch.rand(1) + mu_min
    sigma = (sigma_max - sigma_min) * torch.rand(1) + sigma_min
    precision = sigma ** (-2)

    z = torch.rand(1) + z_min

    return mu, torch.log(precision), torch.log(z)
