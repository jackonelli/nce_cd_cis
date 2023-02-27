import unittest
import torch

from src.models.ace.ace_model import AceModel


class TestBinaryNCE(unittest.TestCase):
    def test_forward(self):

        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)

        num_samples = 100
        input = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                                  scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples, 1))

        observed = torch.distributions.bernoulli.Bernoulli(0.5).sample((num_samples, num_features))
        masked_input = input * observed

        context = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_context_units,))),
                                                    scale=torch.exp(torch.randn(torch.Size((num_context_units,))))).sample((num_samples, num_features))

        unobserved = torch.randint(low=0, high=num_features, size=(num_samples,))

        output = model.log_prob((masked_input, unobserved, context))

        assert output.shape == (num_samples, 1)


if __name__ == "__main__":
    unittest.main()