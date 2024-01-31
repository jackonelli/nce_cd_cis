import torch
from torch import Tensor


def concat_samples(y: Tensor, y_samples: Tensor) -> Tensor:
    """Concatenate y (NxD), y_samples (NxJxD) and to tensor of shape Nx(J+1)xD

    Note that the actual y sample is the first vector of the J+1 concatenated samples.
    """

    return torch.cat((y.reshape(y.shape[0], 1, -1), y_samples), dim=1)


def unnorm_weights(y: Tensor, unnorm_distr, noise_distr) -> Tensor:
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)


def log_unnorm_weights(y: Tensor, log_unnorm_distr, log_noise_distr) -> Tensor:
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return log_unnorm_distr(y) - log_noise_distr(y)


def cond_unnorm_weights(y: Tensor, yp: Tensor, unnorm_distr, noise_distr):
    """Compute weights w(y) = p_tilde_theta(y) / p_n(y| yp) for cond. noise distr.

    Args:
        y (tensor): Sampled y, extended by the actual sample, shape (N, J+1, D)
        yp (tensor): Cond. value, shape (N, D)
        unnorm_distr: unnorm pdf: p: R^NxD -> [0, inf)^N
        noise_distr: pdf: p_n: R^NxD -> [0, inf)^N
    Returns:
        w_tilde (tensor): unnorm. weights for all y, shape (N,)
    """
    return unnorm_distr(y) / noise_distr(y, yp)


def cond_unnorm_weights_ratio(y: Tensor, yp: Tensor, unnorm_distr, noise_distr) -> Tensor:
    """Compute weight ratio w(y)/w(yp) with w(y) = p_tilde_theta(y) / p_n(y| yp) for cond. noise distr.

    Args:
        y (tensor): Sampled y, extended by the actual sample, shape (N, J+1, D)
        yp (tensor): Cond. value, shape (N, D)
        unnorm_distr: unnorm pdf: p: R^NxD -> [0, inf)^N
        noise_distr: pdf: p_n: R^NxD -> [0, inf)^N
    Returns:
        ratio (tensor): unnorm. weights for all y, shape (N,)
    """
    return (
        cond_unnorm_weights(y, yp, unnorm_distr, noise_distr) / cond_unnorm_weights(yp, y, unnorm_distr, noise_distr)
    )


def log_cond_unnorm_weights(
        y: Tensor, yp: Tensor, log_unnorm_distr, log_noise_distr
) -> Tensor:
    """Compute log weights log w(y) = log_p_tilde_theta(y) / log_p_n(y| yp) for cond. noise distr.

    Args:
        y (tensor): Sampled y, extended by the actual sample, shape (N, J+1, D)
        yp (tensor): Cond. value, shape (N, D)
        unnorm_distr: unnorm pdf: p: R^NxD -> [0, inf)^N
        noise_distr: pdf: p_n: R^NxD -> [0, inf)^N
    Returns:
        w_tilde (tensor): log unnorm. weights for all y, shape (N,)
    """

    return (
        log_unnorm_distr(y)
        - log_noise_distr(y, yp)
    )


def log_cond_unnorm_weights_ratio(
        y: Tensor, yp: Tensor, log_unnorm_distr, log_noise_distr
) -> Tensor:
    """Compute log weights ratio log w(y)/w(yp) with w(y) = p_tilde_theta(y) / p_n(y| yp) for cond. noise distr.

    Args:
        y (tensor): Sampled y, extended by the actual sample, shape (N, J+1, D)
        yp (tensor): Cond. value, shape (N, D)
        unnorm_distr: unnorm pdf: p: R^NxD -> [0, inf)^N
        noise_distr: pdf: p_n: R^NxD -> [0, inf)^N
    Returns:
        log_ratio (tensor): log of ratio of unnorm. weights for all y, shape (N,)
    """
    return (
        log_cond_unnorm_weights(y, yp, log_unnorm_distr, log_noise_distr)
        - log_cond_unnorm_weights(yp, y, log_unnorm_distr, log_noise_distr)
    )


def norm_weights(unnorm_weights: Tensor) -> Tensor:
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j) for all y_j"""
    return unnorm_weights / unnorm_weights.sum()


def outer_product(v, h):
    """Compute outer product of 2D matrices"""
    return v.reshape(v.shape[0], -1, 1) * h.reshape(h.shape[0], 1, -1)
