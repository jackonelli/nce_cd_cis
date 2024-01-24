from typing import Optional
import torch
from torch import Tensor


class BaseModel(torch.nn.Module):
    def prob(self, y: Tensor) -> Tensor:
        """Compute unnorm prob p_tilde(y)"""

        return torch.exp(self.log_prob(y))

    def log_prob(self, y: Tensor, x: Optional[Tensor] = None) -> Tensor:
        """Compute log of unnorm prob: log p_tilde(y)"""
        pass

    def forward(self, y: Tensor) -> Tensor:
        return self.prob(y)
        

    def grad_log_prob(self, y: Tensor, weights=torch.tensor(1)):
        """Calculate (weighted) gradient of log probability"""

        # print("Normalised weights", weights)
        l_temp = weights * self.log_prob(y)
        print("Log prob with shape")
        print(l_temp.shape)
        print(l_temp)

        temp_loss = (weights * self.log_prob(y)).sum()
        print(temp_loss)

        print("Grad weights")
        weights.sum().backward()
        gradsw = [param.grad.detach().clone() for param in self.parameters()]

        for i, grad in enumerate(gradsw):
            print("grad w" + str(i))
            print(grad)


        temp_loss.backward()
        grads = [param.grad.detach().clone() for param in self.parameters()]

        for i, grad in enumerate(grads):
            print("grad temp loss " + str(i))
            print(grad)

        return grads

    def set_gradients(self, grads):
        """Manually set parameter gradients"""

        self.clear_gradients()

        for param, grad in zip(self.parameters(), grads):
            param.grad = grad

    def get_gradients(self):
        return [param.grad for param in self.parameters()]

    def clear_gradients(self):
        """Clear all parameter gradients"""

        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
                param.grad.detach_()

    def num_parameters(self):
        """Total number of model parameters"""

        num_params = 0
        for param in self.parameters():
            num_params += param.numel()

        return num_params

