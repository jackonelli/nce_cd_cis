import unittest
import warnings
import torch

from src.models.ace.ace_model import AceModel
from src.training.training_utils import PolynomialLr


class TestPolynomialLR(unittest.TestCase):
    def test_forward(self):
        # Just test so that everything seems to run ok
        warnings.filterwarnings("ignore")

        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)

        lr = 1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Polynomial decaying lr
        num_steps_decay, lr_factor = 10, 0.001
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=PolynomialLr(num_steps_decay, lr,
                                                                             lr * lr_factor).decayed_learning_rate)

        # Check so that lr decreases
        old_lr = scheduler.get_last_lr()
        for i in range(num_steps_decay):
            scheduler.step()
            assert scheduler.get_last_lr() <= old_lr
            old_lr = scheduler.get_lr()

        # Check minimal lr has been reached
        assert scheduler.get_last_lr()[0] == (lr * lr_factor)

        # Check so that lr does not decrease beyond num_steps_decay
        old_lr = scheduler.get_last_lr()
        scheduler.step()
        assert scheduler.get_last_lr() == old_lr


if __name__ == "__main__":
    unittest.main()