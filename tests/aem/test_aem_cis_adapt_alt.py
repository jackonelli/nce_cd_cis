import unittest
import torch

from src.aem.aem_cis_joint_z_adapt_alt import AemCisJointAdaAltCrit
from src.aem.aem_cis_joint_z_adapt import AemCisJointAdaCrit
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class TestAemCisAlt(unittest.TestCase):

    def test_grad(self):
        # Check so that gradient is same as other AEM CIS crit.
        num_features = torch.randint(low=2, high=10, size=torch.Size((1,))).item()
        num_context_units = torch.randint(low=1, high=10, size=torch.Size((1,))).item()
        num_negative = torch.randint(low=1, high=5, size=torch.Size((1,))).item()

        num_res_blocks, num_hidden, num_components = 2, 5, 5
        output_dim_mult = num_context_units + 3 * num_components
        made = ResidualMADEJoint(2 * num_features, num_res_blocks, num_hidden, output_dim_mult)

        model = ResidualEnergyNet(input_dim=(num_context_units + 1))
        proposal = AemJointProposal(made, num_context_units, num_components)

        crit = AemCisJointAdaAltCrit(model, proposal, num_negative)
        crit_ref = AemCisJointAdaCrit(model, proposal, num_negative)

        num_samples = 100
        y = torch.distributions.normal.Normal(loc=torch.randn(torch.Size((num_features,))),
                                              scale=torch.exp(torch.randn(torch.Size((num_features,))))).sample(
            (num_samples,))

        log_q_y, log_q_y_samples_ext, context_y, context_y_samples_ext, y_samples_ext = crit._proposal_log_probs(y,
                                                                                                     num_samples=num_negative, y_sample_base=y)

        # Calculate grads.
        loss, _, _, _ = crit.inner_pers_crit((y, context_y, log_q_y),
                                             (y_samples_ext, context_y_samples_ext, log_q_y_samples_ext))
        loss.backward(retain_graph=True)
        grad_model = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal = [param.grad.detach().clone() for param in proposal.parameters()] # made.parameters()

        # Calculate ref. grads.
        model.clear_gradients()
        proposal.clear_gradients()

        y_samples, context_y_samples, log_q_y_samples = y_samples_ext[y.shape[0]:, :], \
                                                        context_y_samples_ext[y.shape[0]:, ::], \
                                                        log_q_y_samples_ext[y.shape[0]:]
        loss_ref, _, _ = crit_ref.inner_crit((y, context_y, log_q_y), (y_samples, context_y_samples, log_q_y_samples))
        loss_ref.backward()
        grad_model_ref = [param.grad.detach().clone() for param in model.parameters()]
        grad_proposal_ref = [param.grad.detach().clone() for param in proposal.parameters()]

        # Check parameter grads same
        for p, p_ref in zip(grad_model, grad_model_ref):
            assert torch.max(torch.abs(p-p_ref)) < 1e-5
            assert torch.allclose(p, p_ref, atol=1e-5, rtol=1e-3)

        for p, p_ref in zip(grad_proposal, grad_proposal_ref):
            assert torch.max(torch.abs(p-p_ref)) < 1e-5
            assert torch.allclose(p, p_ref, atol=1e-5, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
