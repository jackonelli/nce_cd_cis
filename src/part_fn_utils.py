import torch
from torch import Tensor


def unnorm_weights(y: Tensor, unnorm_distr, noise_distr) -> Tensor:
    """Compute w_tilde(y) = p_tilde(y) / p_n(y)"""
    return unnorm_distr(y) / noise_distr(y)


def cond_unnorm_weights(y: Tensor, yp: Tensor, unnorm_distr, noise_distr) -> Tensor:
    return (
        unnorm_distr(y) * noise_distr(yp, y) / (unnorm_distr(yp) * noise_distr(y, yp))
    )


def log_cond_unnorm_weights(
    y: Tensor, yp: Tensor, log_unnorm_distr, log_noise_distr
) -> Tensor:
    return (
        log_unnorm_distr(y)
        + log_noise_distr(yp, y)
        - log_unnorm_distr(yp)
        - log_noise_distr(y, yp)
    )


def norm_weights(unnorm_weights: Tensor) -> Tensor:
    """Compute self-normalised weight w(y) = w_tilde(y) / sum_j w_tilde(y_j) for all y_j"""
    return unnorm_weights / unnorm_weights.sum()


def concat_samples(y: Tensor, y_samples: Tensor) -> Tensor:
    """Concatenate y (NxD), y_samples (NxJxD) and to tensor of shape Nx(J+1)xD"""

    return torch.cat((y.reshape(y.shape[0], 1, -1), y_samples), dim=1)
