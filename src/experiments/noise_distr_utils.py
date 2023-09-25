import torch
import numpy as np

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal
from src.nce.cnce import CondNceCrit


# Noise distribution parameters, NCE
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gNCEnoise.m
def get_nce_noise_distr_par(y, eps=1e-10):
    """ Parameters of unconditional noise distr. (Gaussian)
    :param y: ((N, D) tensor) training data
    :param eps: (float) small constant, needed to ensure positive eigenvalues of cov. matrix
    :return: ((D,) tensor) mean, ((D, D) tensor) covariance matrix of unconditional noise distr.
    """
    mu = torch.mean(y, dim=0)
    cov = torch.cov(torch.transpose(y, 0, 1))

    if not (cov == cov.T).all():
        cov = (cov + cov.T) / 2

    #   if not (torch.eig(cov_noise_nce)[0][:,0] >= 0).all():
    cov = cov + eps * torch.eye(cov.shape[0])

    return mu, cov


# Noise distribution parameters, CNCE
# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gNoise.m
def get_cnce_epsilon_base(y):
    """ Calculate basis for covariance matrix in conditional noise distr. (Gaussian)
    :param y: ((N, D) tensor) training data
    :return: mean standard deviation of training data
    """
    return torch.std(y, dim=-1).mean()


def get_cnce_covariance_matrix(epsilon_factor, epsilon_base, num_dims):
    """ Calculate (diagonal) covariance matrix of conditional noise distr. (Gaussian)
    :param epsilon_factor: (float) scaling factor for variance
    :param epsilon_base: (float) basis for variance of covariance matrix
    :num_dims: (int) number of data dimensions (D)
    :return: ((D, D) tensor) diagonal covariance matrix
    """
    return torch.eye(num_dims) * (epsilon_factor * epsilon_base) ** 2


# Adapted from https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/noise/continous/gEpsilonBaseFun.m
def get_cnce_noise_distr_par(y, neg_sample_size, p_m):
    """ Calculate covariance matrix for conditional noise distr. (Gaussian)
    :param y: ((N, D) tensor) training data
    :param neg_sample_size: (int) number of negative samples used for training
    :param p_m: (BaseModel) model
    :return: ((D, D) tensor) (diagonal) covariance matrix for conditional noise distr.
    """
    epsilon_factor = get_cnce_epsilon_factor(y, neg_sample_size, p_m)
    epsilon_base = get_cnce_epsilon_base(y)
    return get_cnce_covariance_matrix(epsilon_factor, epsilon_base, y.size(-1))


def evaluate_cnce_loss(epsilon_factor, y, neg_sample_size, p_m):
    """ Evaluate CNCE loss (for given scaling factor of covariance matrix)
    :param y: ((N, D) tensor) training data
    :param epsilon_factor: (float) scaling factor for covariance matrix of (Gaussian) noise distr.
    :param neg_sample_size: (int) number of negative samples used for training
    :param p_m: (BaseModel) model
    :return: (float) value of CNCE loss
    """
    epsilon_base = get_cnce_epsilon_base(y)
    cov_noise = get_cnce_covariance_matrix(epsilon_factor, epsilon_base, y.size(-1))

    p_n = ConditionalMultivariateNormal(cov=cov_noise)
    criterion = CondNceCrit(p_m, p_n, neg_sample_size)

    return criterion.crit(y, 0)


def get_cnce_epsilon_factor(y, neg_sample_size, p_m, thrs_lower=0.05, thrs_upper=0.5,
                            inc_rate=0.2, dec_rate=0.5, max_iter=500, eps_hard_cap=1000):
    """Find scaling factor for covariance matrix of conditional noise distr. (Gaussian)
    :param y: ((N, D) tensor) training data
    :param neg_sample_size: (float) number of negative samples used in training
    :param p_m: (BaseModel) model
    :param thrs_lower: (float) lower threshold on loss
    :param thrs_upper: (float) (factor for) upper threshold on loss
    :param inc_rate: (float) increase rate of scaling factor
    :param dec_rate: (float) decrease rate of scaling factor
    :param max_iter: (int) maximum number of iterations
    :param eps_hard_cap: (float) maximum value for scaling factor
    :return (float) scaling factor for covariance matrix of conditional noise distr.
    """

    loss_zero = np.log(2)  # Loss as epsilon -> 0
    loss_inf = 0  # Loss as epsilon -> inf
    thrs_upper = thrs_upper * loss_zero

    epsilon_factor = 0.5  # Start value

    # Calculate initial loss
    loss = evaluate_cnce_loss(epsilon_factor, y, neg_sample_size, p_m)

    # Iterate until conditions are met
    k = 1
    while (k < max_iter) and (abs(1 - (loss / loss_zero)) < thrs_lower or loss < thrs_upper) and (
            epsilon_factor < eps_hard_cap):

        if abs(1 - (loss / loss_zero)) < thrs_lower:
            epsilon_factor = (1 + inc_rate) * epsilon_factor
        elif loss < thrs_upper:
            epsilon_factor = (1 - dec_rate) * epsilon_factor

        loss = evaluate_cnce_loss(epsilon_factor, y, neg_sample_size, p_m)

        k = k + 1

    return epsilon_factor