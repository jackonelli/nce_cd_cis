# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch
from torch.distributions import Categorical

from src.aem.aem_smc import AemSmcCrit
from src.noise_distr.aem_proposal_joint_z import AemJointProposal


class AemSmcCondCrit(AemSmcCrit):
    def __init__(self, unnorm_distr, noise_distr: AemJointProposal, num_neg_samples: int,
                 num_neg_samples_validation: int=1e2, alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def crit(self, y, _idx):
        return self.inner_crit(y)

    def inner_crit(self, y, y_samples=None):

        y = y[0]
        assert y.ndim == 2
        assert y.shape[0] > 1

        # Calculate (unnormalized) densities for y
        log_q_y, context = torch.zeros((y.shape[0], self.dim)), torch.zeros((y.shape[0], self.dim, self.num_context_units))
        for i in range(self.dim):
            log_q_y[:, i], context[:, i, :], _ = self._proposal_log_probs(y, i, num_observed=y.shape[0])

        log_p_tilde_y = self._model_log_probs(y.reshape(-1, 1), context.reshape(-1, self.num_context_units))

        # Estimate log normalizer + expected log q
        log_normalizer, log_q = self.smc(y.shape[0], y=y)

        # Calculate loss
        p_loss = - torch.mean(torch.sum(log_p_tilde_y, dim=-1) - log_normalizer)
        q_loss = - torch.mean(log_q)

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss

    def smc(self, batch_size, y=None):

        if y is not None:
            assert batch_size == y.shape[0]
            num_observed = batch_size
            num_chains = self._num_neg + 1
            y_s = torch.cat((y.unsqueeze(dim=1), torch.zeros((batch_size, self._num_neg, self.dim))), dim=1)
        else:
            num_observed = 0
            num_chains = self._num_neg
            y_s = torch.zeros((batch_size, num_chains, self.dim))

        log_q_y_s = torch.zeros((batch_size, num_chains, self.dim))

        # First dim
        # Propagate
        log_q_y_s[:, :, 0], context, y_s = self._proposal_log_probs(y_s.reshape(-1, 1), 0, num_observed=num_observed)

        # Reweight
        log_p_tilde_y_s = self._model_log_probs(y_s[:, :, 0].reshape(-1, 1),
                                                context.reshape(-1, self.num_context_units))
        log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_chains, self.dim)

        # Dim 2 to D
        log_normalizer = torch.tensor(0.0)
        for i in range(1, self.dim):
            # Resample
            with torch.no_grad():
                ancestor_inds = Categorical(logits=log_w_tilde_y_s).sample(sample_shape=torch.Size(self._num_neg + 1, ))

            y_s[:, :, :i - 1] = torch.gather(y_s[:, :, :i - 1], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i))
            log_q_y_s[:, :, :i - 1] = torch.gather(log_q_y_s[:, :, :i - 1], dim=1, index=ancestor_inds[:, :, None].repeat(1, 1, i))

            # Propagate
            log_q_y_s[:, :, i], context, y_s = self._proposal_log_probs(y_s.reshape(-1, 1), i, num_observed=num_observed)

            # Reweight
            log_p_tilde_y_s = self._model_log_probs(y_s[:, :, i].reshape(-1, 1),
                                                    context.reshape(-1, self.num_context_units))
            log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s[:, :, i].detach()).reshape(-1, num_chains)  # TODO: övriga termer tar ut varandra?
            log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor(num_chains))

        log_q = torch.exp(log_normalizer) * torch.sum(torch.nn.Softmax(dim=-1)(log_w_tilde_y_s)
                                                      * torch.sum(log_q_y_s, dim=-1), dim=-1)

        return log_normalizer, log_q

