"""Conditional Restricted Boltzmann Machine"""
import torch
from torch import Tensor

from src.models.base_model import BaseModel


class CondRbm(BaseModel):

    def __init__(self, weights, vis_bias, hidden_bias, class_weights):#, class_bias):
        super().__init__()

        self.weights = torch.nn.Parameter(weights, requires_grad=True)
        self.vis_bias = torch.nn.Parameter(vis_bias, requires_grad=True)
        self.hidden_bias = torch.nn.Parameter(hidden_bias, requires_grad=True)
        self.class_weights = torch.nn.Parameter(class_weights, requires_grad=True)
        #self.class_bias = torch.nn.Parameter(class_bias, requires_grad=True)

    def log_prob(self, y: tuple) -> Tensor:
        v, x = y
        return - self.energy(v, x)

    def energy(self, v: Tensor, x: Tensor):
        # From http: // swoh.web.engr.illinois.edu / courses / IE598 / handout / rbm.pdf

        logits_p_h = self.hidden_model(v, x)
        return - ((v * self.filtered_vis_bias(x)).sum(dim=-1)
                  + torch.logsumexp(torch.cat((torch.zeros(logits_p_h.shape), logits_p_h), dim=0), dim=0).sum(dim=-1))

    def total_energy(self, v: Tensor, h: Tensor, x: Tensor):
        return - ((h * self.filtered_class_weights(x) * torch.matmul(v, self.weights)).sum(dim=-1) \
               + (v * self.filtered_vis_bias(x)).sum(dim=-1) - (h * self.filtered_hidden_bias(x)).sum(dim=-1))


    def filtered_class_weights(self, x: Tensor):
        return torch.matmul(x, self.class_weights) #+ self.class_bias.t()

    def filtered_vis_bias(self, x: Tensor):
        return torch.matmul(x, self.vis_bias)

    def filtered_hidden_bias(self, x: Tensor):
        return torch.matmul(x, self.hidden_bias)

    def hidden_model(self, v: Tensor, x: Tensor):
        return self.filtered_class_weights(x) * torch.matmul(v, self.weights) + self.filtered_hidden_bias(x)

    def visible_model(self, h: Tensor, x: Tensor):
        if x.ndim >= 3:
            num_samples = x.shape[0]
            return torch.bmm(self.filtered_class_weights(x.reshape(-1, x.shape[-1])).unsqueeze(dim=1)
                             * self.weights.unsqueeze(dim=0),
                             h.reshape(-1, h.shape[-1]).unsqueeze(dim=-1)).reshape(
                num_samples, -1, self.weights.shape[0]) + self.filtered_vis_bias(x)

        else:
            return torch.bmm(self.filtered_class_weights(x).unsqueeze(dim=1) * self.weights.unsqueeze(dim=0),
                             h.unsqueeze(dim=-1)).squeeze(dim=-1) + self.filtered_vis_bias(x)

    def sample_hidden(self, v: Tensor, x: Tensor):
        z = self.hidden_model(v, x)
        p_h = torch.sigmoid(z)
        h = torch.distributions.bernoulli.Bernoulli(logits=z).sample()

        return p_h, h

    def sample_visible(self, h: Tensor, x: Tensor):
        z = self.visible_model(h, x)
        p_v = torch.sigmoid(z)
        v = torch.distributions.bernoulli.Bernoulli(logits=z).sample()

        return p_v, v

    def sample_from_hidden(self, h: Tensor, x: Tensor, k=1):
        """Sample from distribution given label y"""

        p_v, v = self.sample_visible(h, x)
        for _ in range(k - 1):
            _, h = self.sample_hidden(v, x)
            p_v, v = self.sample_visible(h, x)

        return p_v, v

    def sample(self, v: Tensor, x: Tensor, k=1):
        """Sample from distribution"""

        p_v, v = v.clone(), v.clone()
        for _ in range(k):
            _, h = self.sample_hidden(v, x)
            p_v, v = self.sample_visible(h, x)

        return p_v, v


