"""Metrics"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


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
