import unittest
import torch

from src.nce.ace_is import AceIsCrit
from src.models.ace.ace_model import AceModel, ResidualBlock
from src.noise_distr.ace_proposal import AceProposal

from src.experiments.ace_exp_utils import UniformMaskGenerator


class TestAceIs(unittest.TestCase):
    def test_crit(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceIsCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample((num_samples,))

        loss, _, _ = crit.crit(y, 0)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss) or torch.isinf(loss)

    def test_grad(self):
        # Check so that gradients of both proposal and model are updated in gradient update
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceIsCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        # Update only for model
        crit.calculate_crit_grad_p(y, 0)

        # Check that parameters of model have been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            else:
                print("Note: param is not torch.nn.Parameter")

        # Check that parameters of proposal have NOT been updated
        # TODO: should it be updated based on context??
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                pass
                #assert torch.allclose(param.grad, torch.tensor(0.0))
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

        # Update only for proposal
        model.clear_gradients()
        crit.calculate_crit_grad_q(y, 0)

        # Check that parameters of model have NOT been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert torch.allclose(param.grad, torch.tensor(0.0))
            else:
                print("Note: param is not torch.nn.Parameter")

        # Check that parameters of proposal have been updated
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

        # Update for both model and proposal
        model.clear_gradients()
        crit.calculate_crit_grad(y, 0)

        # Check that parameters of model have been updated
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            else:
                print("Note: param is not torch.nn.Parameter")

            # Check that parameters of proposal have been updated
        for param in proposal.parameters():
            if isinstance(param, torch.nn.Parameter):
                assert param.grad is not None
            if isinstance(param, ResidualBlock):
                print("Note: param is not torch.nn.Parameter")

    def test_perm_mask(self):
        # Test so that permutation for log-likelihood calculation works as intended
        observed_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        y = torch.tensor([[5.0, 7.3, 2.0], [-1.0, 2.2, 4.5]])
        model, proposal = None, None

        crit = AceIsCrit(model, proposal, num_neg_samples=1)

        expanded_y, expanded_observed_mask, selected_features = [], [], []
        for i in range(y.shape[0]):
            y_e, mask, sf = crit._permute_mask(y[i, :], observed_mask[i, :])
            expanded_y.append(y_e)
            expanded_observed_mask.append(mask)
            selected_features.append(sf)

        expanded_y, expanded_observed_mask, selected_features = torch.cat(expanded_y, dim=0), \
                                                                torch.cat(expanded_observed_mask, dim=0), \
                                                                torch.cat(selected_features, dim=0)

        num_queries = torch.count_nonzero(observed_mask)
        assert expanded_y.shape == (num_queries, y.shape[-1])
        assert expanded_observed_mask.shape == (num_queries, y.shape[-1])
        assert selected_features.shape == (num_queries, 1)

        expanded_y_ref = torch.tensor([[5.0, 7.3, 2.0], [-1.0, 2.2, 4.5], [-1.0, 2.2, 4.5]])
        assert torch.allclose(expanded_y, expanded_y_ref)

        expanded_observed_mask_ref_1 = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        expanded_observed_mask_ref_2 = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        assert torch.allclose(expanded_observed_mask, expanded_observed_mask_ref_1) or \
               torch.allclose(expanded_observed_mask, expanded_observed_mask_ref_2)

        if torch.allclose(expanded_observed_mask, expanded_observed_mask_ref_1):
            selected_features_ref = torch.tensor([1, 0, 1]).reshape(-1, 1)
            assert torch.allclose(selected_features, selected_features_ref)
        elif torch.allclose(expanded_observed_mask, expanded_observed_mask_ref_2):
            selected_features_ref = torch.tensor([1, 1, 0]).reshape(-1, 1)
            assert torch.allclose(selected_features, selected_features_ref)

    def test_generate_model_input(self):
        # Test generating model input when selected features is not None
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        model, proposal = None, None

        crit = AceIsCrit(model, proposal, num_negative, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        y_samples = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples, num_negative))

        context = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_context_units,))),
                                                    scale=torch.exp(torch.randn(torch.Size((num_context_units,))))).sample(
            (num_samples, num_features))

        selected_features = torch.randint(low=0, high=num_features, size=(num_samples, 1))

        unobserved_mask = UniformMaskGenerator()(num_samples, num_features)

        y_u, y_samples_u = y * unobserved_mask, y_samples * unobserved_mask.unsqueeze(dim=1)
        y_u_i, u_i, tiled_context = crit._generate_model_input(y_u, y_samples_u, context, selected_features)

        assert y_u_i.shape == (y.shape[0] * (num_negative + 1),)
        assert u_i.shape == (y.shape[0] * (num_negative + 1),)
        assert tiled_context.shape == (y.shape[0] * (num_negative + 1), num_context_units)

        y_u_i_ref, u_i_ref, tiled_context_ref = [], [], []
        y_s = torch.cat((y_u.unsqueeze(dim=1), y_samples_u), dim=1)
        for i in range(y.shape[0]):
            y_u_i_ref.append(y_s[i, :, selected_features[i]])
            u_i_ref.append(torch.tile(selected_features[i], (num_negative + 1,)))
            tiled_context_ref.append(torch.tile(context[i, selected_features[i], :], (1 + num_negative, 1)))

        assert torch.allclose(y_u_i, torch.cat(y_u_i_ref, dim=0).reshape(-1))
        assert torch.allclose(u_i, torch.cat(u_i_ref, dim=0).reshape(-1))
        assert torch.allclose(tiled_context, torch.cat(tiled_context_ref, dim=0).reshape(-1, num_context_units))

    def test_log_likelihood(self):
        # Just test so that everything seems to run ok
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_is_samples = torch.randint(low=10, high=50, size=torch.Size((1,))).item()

        model = AceModel(num_features=num_features, num_context_units=num_context_units)
        proposal = AceProposal(num_features=num_features, num_context_units=num_context_units)
        crit = AceIsCrit(model, proposal, 1, energy_reg=0.1)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        ll = crit.log_likelihood(y, num_is_samples)
        assert ll.shape == torch.Size([1])
        assert not torch.isnan(ll) or torch.isinf(ll)


if __name__ == "__main__":
    unittest.main()
