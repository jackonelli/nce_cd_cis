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


# In CNCE article they use https://github.com/ciwanceylan/CNCE-matlab/blob/master/matlab/natural_images/bin/estimation/optimisation/minimize.m
# or fminunc in MATLAB
def no_change_stopping_condition(new_params, old_params, tol=1e-6):

    step_condition = (torch.abs(new_params - old_params) / (1 + torch.abs(old_params))).mean() < tol
    if step_condition:
        return True
    else:
        return False