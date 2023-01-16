import torch


def initialise_rbm_params(num_visible=784, num_hidden=500):
    """Initialise parameters of RBM model"""

    weights = torch.nn.Parameter(torch.randn(num_visible, num_hidden) * 1e-2)
    vis_bias = torch.nn.Parameter(torch.zeros((num_visible, 1)))
    hidden_bias = torch.nn.Parameter(torch.zeros((num_hidden, 1)))

    return weights, vis_bias, hidden_bias


def initialise_cond_rbm_params(num_visible=784, num_hidden=500, num_conditional=10):
    """Initialise parameters of RBM model"""

    """Initialise parameters of RBM model"""

    weights = torch.nn.Parameter(torch.randn(num_visible, num_hidden) * 1e-2)
    vis_bias = torch.nn.Parameter(torch.zeros((num_conditional, num_visible)))
    hidden_bias = torch.nn.Parameter(torch.zeros((num_conditional, num_hidden)))

    class_weights = torch.nn.Parameter(torch.randn(num_conditional, num_hidden) * 1e-2)
    class_bias = torch.nn.Parameter(torch.zeros((num_hidden, 1)))

    return weights, vis_bias, hidden_bias, class_weights #, class_bias


def initialise_ebm_params(num_visible=784, num_hidden=500):
    """Initialise parameters of RBM model"""

    input_weights = torch.nn.Parameter(torch.randn(num_visible, num_hidden) * 1e-2)
    input_bias = torch.nn.Parameter(torch.zeros((num_hidden, 1)))
    hidden_weights = torch.nn.Parameter(torch.randn(num_hidden, 1) * 1e-2)
    hidden_bias = torch.nn.Parameter(torch.zeros((1, 1)))

    return input_weights, input_bias, hidden_weights, hidden_bias


def initialise_cond_ebm_params(num_visible=784, num_hidden=500, num_conditional=10):
    """Initialise parameters of RBM model"""

    input_weights = torch.nn.Parameter(torch.randn((num_visible, num_hidden)) * 1e-2)
    input_bias = torch.nn.Parameter(torch.zeros((num_conditional, num_hidden)))
    hidden_weights = torch.nn.Parameter(torch.randn(num_hidden, 1) * 1e-2)
    hidden_bias = torch.nn.Parameter(torch.zeros((num_conditional, 1)))

    input_class_weights = torch.nn.Parameter(torch.randn((num_conditional, num_hidden)) * 1e-2)
    input_class_bias = torch.nn.Parameter(torch.zeros((num_conditional, num_visible)))
    hidden_class_weights = torch.nn.Parameter(torch.randn((num_conditional, 1)) * 1e-2)
    hidden_class_bias =torch.nn.Parameter(torch.zeros((num_conditional, num_hidden)))

    return input_weights, input_bias, hidden_weights, hidden_bias, input_class_weights, input_class_bias, \
           hidden_class_weights, hidden_class_bias






