# Adapted from https://github.com/lupalab/ace/blob/main/ace/masking.py
from abc import ABC, abstractmethod
import numpy as np
import torch


class MaskGenerator(ABC):
    def __init__(
        self,
        seed=1,
    ):
        self.gen = torch.Generator().manual_seed(seed)    #device="cuda"

    def __call__(self, num_samples, num_features):
        return self.call(num_samples, num_features)

    @abstractmethod
    def call(self, num_samples, num_features):
        pass


class UniformMaskGenerator(MaskGenerator):
    def call(self, num_samples, num_features):

        # For each obs., observe 0 to num_features-1 features
        k = torch.randint(low=0, high=num_features-1, size=(num_samples,), generator=self.gen)

        result = []
        for i in range(num_samples):
            mask = torch.zeros(num_features)
            inds = torch.randperm(num_features, generator=self.gen)[:k[i]]
            mask[inds] = 1
            result.append(mask)

        return torch.vstack(result)


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, num_samples, num_features):
        return torch.bernoulli(torch.tensor([self.p] * num_samples * num_features),
                               generator=self.gen).reshape(num_samples, num_features)