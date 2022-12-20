import torch


def initialise_params(num_visible=784, num_hidden=500):

    weights = torch.nn.Parameter(torch.randn(num_visible, num_hidden) * 1e-2)
    vis_bias = torch.nn.Parameter(torch.zeros(num_visible))
    hidden_bias = torch.nn.Parameter(torch.zeros(num_hidden))

    return weights, vis_bias, hidden_bias