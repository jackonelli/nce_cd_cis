import unittest
import torch

from src.models.ace.ace_model import AceModel


class TestAceModel(unittest.TestCase):
    def test_forward(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)

        num_samples = 100
        input = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                                  scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        observed = torch.distributions.bernoulli.Bernoulli(0.5).sample((num_samples, num_features))
        masked_input = input * observed

        context = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_context_units,))),
                                                    scale=torch.exp(torch.randn(torch.Size((num_context_units,))))).sample((num_samples,))

        unobserved = torch.broadcast_to(torch.arange(0, num_features, dtype=torch.int64), [num_samples, num_features])

        output = model.log_prob((masked_input.reshape(-1), unobserved.reshape(-1),
                                 torch.tile(context.unsqueeze(dim=1), [1, num_features, 1]).reshape(-1, num_context_units)))

        assert output.shape == (num_samples * num_features, 1)


if __name__ == "__main__":
    unittest.main()