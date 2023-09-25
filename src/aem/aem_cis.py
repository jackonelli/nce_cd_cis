# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import torch

from src.aem.aem_is import AemIsCrit
from src.noise_distr.aem_proposal import AemProposal


class AemCisCrit(AemIsCrit):
    def __init__(self, unnorm_distr, noise_distr: AemProposal, num_neg_samples: int, num_neg_samples_validation: int=1e2,
                 alpha: float = 1.0):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples, num_neg_samples_validation, alpha)

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        y, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples =\
            self._log_probs(y, y_samples, context, q, self._num_neg)

        # calculate log normalizer
        log_w_tilde_y = (log_p_tilde_y - log_q_y.detach())
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach())
        log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples), dim=1)
        assert log_w_tilde_y_s.shape == (y.shape[0], 1 + self._num_neg, y.shape[-1])
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1]))

        # calculate normalized density
        p_loss = - torch.mean(torch.sum(
            log_p_tilde_y - log_normalizer,
            dim=-1
        ))
        q_loss = - torch.mean(torch.sum(
            log_q_y,
            dim=-1
        ))

        loss = q_loss + self.alpha * p_loss

        return loss, p_loss, q_loss

    def log_prob(self, y, conditional_inputs=None):

        q, context = self._noise_distr.forward(y, conditional_inputs)

        if self._noise_distr.Component is not None:
            y_samples = self.inner_sample_noise(q, num_samples=self.num_neg_samples_validation)
        else:
            y_samples = self.inner_sample_noise(q, num_samples=self.num_neg_samples_validation * y.shape[0] * y.shape[1]
                                                ).reshape(y.shape[0], self.num_neg_samples_validation, y.shape[1])

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples = \
            self._log_probs(y, y_samples, context, q, self.num_neg_samples_validation)

        # calculate log normalizer
        log_w_tilde_y = (log_p_tilde_y - log_q_y.detach())
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach())
        log_w_tilde_y_s = torch.cat((log_w_tilde_y.unsqueeze(dim=1), log_w_tilde_y_samples), dim=1)
        assert log_w_tilde_y_s.shape == (y.shape[0], 1 + self._num_neg, y.shape[-1])
        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([self._num_neg + 1]))

        # calculate normalized density
        log_prob_p = torch.sum(log_p_tilde_y - log_normalizer, dim=-1)
        log_prob_q = torch.sum(log_q_y, dim=-1)

        return log_prob_p, log_prob_q


