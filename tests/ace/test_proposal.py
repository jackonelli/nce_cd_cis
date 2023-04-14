import unittest
import torch

from src.noise_distr.ace_proposal import AceProposal


class TestAceProposal(unittest.TestCase):
    def test_forward(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()

        model = AceProposal(num_features=num_features, num_context_units=num_context_units)

        num_samples = 100
        input = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                                  scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        observed = torch.distributions.bernoulli.Bernoulli(0.5).sample((num_samples, num_features))
        masked_input = input * observed

        distr, context = model.forward((masked_input, observed))

        assert context.shape == (num_samples, num_features, num_context_units)

        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()
        samples = model.inner_sample(distr, torch.Size((num_negative,)))

        print(distr.mixture_distribution)


        assert samples.shape == (num_samples, num_negative, num_features)

        log_prob = model.inner_log_prob(distr, samples[:, 0, :])
        assert log_prob.shape == (num_samples, num_features)


if __name__ == "__main__":
    unittest.main()