import torch


# Metrices
class PrecisionErrorMetric:
    def __init__(self, true_precision):
        self.true_precison = true_precision

    def metric(self, model):
        return torch.mean((torch.exp(model.log_precision) - self.true_precison) ** 2)


# Stopping conditions
def no_stopping(new_params, old_params):
    return False


def no_change_stopping_condition(new_params, old_params, tol=1e-3):

    if torch.abs(new_params - old_params).mean() < tol:
        return True
    else:
        return False