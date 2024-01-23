"""Experiment utilities"""
from pathlib import Path
from typing import List
import torch
from torch import Tensor
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np


def generate_bounds(data, lower_pctl=25, upper_pctl=75):
    assert lower_pctl < upper_pctl
    median, lower, upper = (
        np.median(data, axis=0),
        np.percentile(data, lower_pctl, axis=0),
        np.percentile(data, upper_pctl, axis=0),
    )
    return median, lower, upper


def table_data(median, lower, upper):
    iters = np.arange(1, len(median) + 1)
    err_low = median - lower
    err_upp = upper - median
    return iters, median, err_low, err_upp


def skip_list_item(lst, nth: int):
    """Skip every nth element in list"""
    return list(
        map(
            lambda val: val[1],
            filter(lambda idx_row: (idx_row[0] % nth == 0), enumerate(lst)),
        )
    )


def format_table(iters, median, err_low, err_upp, header):
    """Format tikz table

    Args:
        data: rows with samples
        header: list of strings describing the columns (e.g. x, y, z)
    """
    str_ = [",".join(header) + "\n"]
    for it, med, low, upp in zip(iters, median, err_low, err_upp):
        str_.append(f"{it},{med},{low},{upp}\n")
    return str_


def process_plot_data(data: Tensor, max_iters: int, res: int):
    """Generate seq. for x-axis and modify resolution
    Args:
        data: (N, D) N data points, D columns (e.g. D=3: p_d(x), p_theta(x), q_phi(x))
        max_iters: must be smaller than N.
        res: number of steps between data points.
    """
    iters = torch.arange(start=0, end=max_iters, step=res)
    data = data[iters, :]
    data = torch.column_stack((iters, data))
    data[:, 0] += 1
    return data


def nan_pad(data: Tensor, length: int):
    """Pad tensor with NaN

    Args:
        data: shape (N, )
        length: length >= N
    """
    assert data.dim() == 1, "Expects 1D array, with shape (N, )"
    N = data.size(0)
    assert length >= N, "Padding length must be larger than data length"
    padded = torch.empty((length,))
    padded[:N] = data
    padded[N:] = torch.nan
    return padded


def mvn_curve(mu, cov, std=1, res=100):
    with torch.no_grad():
        angles = torch.linspace(0, 2 * torch.pi, res)
        curve_param = torch.column_stack((torch.cos(angles), torch.sin(angles)))
        ellipsis = std * curve_param @ torch.Tensor(sqrtm(cov))
        return mu + ellipsis


def plot_mvn(levels, ax, label):
    ax.plot(levels[:, 0], levels[:, 1], label=label)


def plot_distrs_ideal(p_d, p_t_d, p_t_t):
    _, ax = plt.subplots()
    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    distrs = [
        (p_d.mu, p_d.cov, "$p_{d}}$"),
        (p_t_d.mu, p_t_d.cov(), "$q=p_d$"),
        (p_t_t.mu, p_t_t.cov(), "$q = p_{\\theta}$"),
    ]
    for mu, cov, label in distrs:
        plot_mvn(mvn_curve(mu, cov), ax, label)
    ax.set_title("Comparison, optimal proposal distrs.")
    ax.legend()


def plot_distrs_adaptive(p_d, p_theta, q_phi):
    _, ax = plt.subplots()
    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    distrs = [
        (p_d.mu, p_d.cov, "$p_{d}}$"),
        (p_theta.mu, p_theta.cov(), "$p_{\\theta}$"),
        (q_phi.mu, q_phi.cov(), "$q_{\\varphi}$"),
    ]
    for mu, cov, label in distrs:
        plot_mvn(mvn_curve(mu, cov), ax, label)
    ax.set_title("Adaptive proposal")
    ax.legend()

def wasserstein_metric(
    x_samples: np.ndarray, y_samples: np.ndarray, p: float = 2.0
) -> float:
    """Wasserstein metric

    Computes the empirical Wasserstein p-distance between x_samples and y_samples
    by solving a linear assignment problem.

    Args:
        x_samples: samples
        y_samples: samples
        p: [0, inf) type of Wasserstein distance
    """

    d = cdist(x_samples, y_samples) ** p
    assignment = linear_sum_assignment(d)
    dist = (d[assignment].sum() / len(assignment)) ** (1.0 / p)
    return dist.item()
