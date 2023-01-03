import torch


def initialise_params(num_visible=784, num_hidden=500):
    """Initialise parameters of RBM model"""

    weights = torch.nn.Parameter(torch.randn(num_visible, num_hidden))
    vis_bias = torch.nn.Parameter(torch.zeros((num_visible, 1)))
    hidden_bias = torch.nn.Parameter(torch.zeros((num_hidden, 1)))

    return weights, vis_bias, hidden_bias

