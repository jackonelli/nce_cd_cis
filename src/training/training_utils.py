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


# From https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/synthetic_data/bin/estimation/cnce.m
def no_change_stopping_condition(new_params, old_params, tol=1e-4):

    step_condition = torch.sqrt(torch.sum((new_params - old_params)**2)) < torch.sqrt(torch.sum(new_params**2)) * tol
    if step_condition:
        return True
    else:
        return False