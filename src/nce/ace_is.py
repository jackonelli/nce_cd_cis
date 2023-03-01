"""Noise Contrastive Estimation (NCE) ranking partition functions with multiple MCMC steps"""
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import log_unnorm_weights, concat_samples

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel
from src.experiments.ace_experiment_utils import UniformMaskGenerator


class AceIsCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = 1 # For now, this option is not available
        self.alpha = alpha  # For regularisation
        self.energy_reg = energy_reg # TODO: In other code, they assign this to the data
        self.mask_generator = UniformMaskGenerator()  # TODO: set seed?

    def crit(self, y: Tensor, _idx: Optional[Tensor]):
        # Mask input
        y_o, y_u, observed_mask = self._mask_input(y)

        q, context = self._noise_distr.forward((y_o, observed_mask))
        y_samples = self.inner_sample_noise(q, num_samples=self._num_neg).detach().clone()

        return self.inner_crit((y_o, y_u, observed_mask, context), (y_samples, q))

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        # Note that we calculate the criterion and not the gradient directly
        # Note: y, y_samples are tuples
        # TODO: For persistance (y_base is not None), it might be easier to ass y_base also to ys

        # Mask input
        y_o, y_u, observed_mask, context = y
        y_samples, q = y_samples

        y_samples_u = y_samples * (1 - observed_mask).unsqueeze(dim=1)

        # Calculate log prob for proposal
        log_q_y = self._noise_distr.inner_log_prob(q, y_u) * (1 - observed_mask)
        log_q_y_samples = self._noise_distr.inner_log_prob(q, y_samples.transpose(0, 1)).transpose(0, 1)
        log_q_y_samples *= (1 - observed_mask).unsqueeze(dim=1)

        q_mean = q.mean * (1 - observed_mask)

        # Calculate log prob for model
        y_u_i, u_i, tiled_context = self._generate_model_input(y_u, y_samples_u, context) # TODO: detach på context??
        log_p_tilde_ys = self._unnorm_distr.log_prob((y_u_i, u_i, tiled_context)).reshape(-1, self._num_neg + 1, y_o.shape[-1])
        log_p_tilde_ys *= (1 - observed_mask).unsqueeze(dim=1)

        assert log_p_tilde_ys.shape[0] == y_o.shape[0]

        # Calculate weights
        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_ys[:, 0, :], log_p_tilde_ys[:, 1:, :]  # TODO: not last col?

        log_w_tilde_y = log_p_tilde_y - log_q_y.detach().clone()
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)

        log_z = (torch.logsumexp(log_w_tilde_y_samples, dim=1) - torch.log(torch.tensor(self._num_neg))) * (1 - observed_mask)

        log_p_y = log_p_tilde_y - log_z
        is_weights = torch.nn.Softmax(dim=-1)(log_w_tilde_y_samples)
        energy_mean = torch.sum(is_weights * y_samples_u) * (1 - observed_mask)

        p_loss = - self.alpha * torch.mean(torch.sum(log_p_y, dim=-1))
        q_loss = - torch.mean(torch.sum(log_q_y, dim=-1))

        if self.energy_reg != 0.0:
            p_loss += self.energy_reg * torch.nn.MSELoss()(log_p_y, log_q_y.detach().clone())

        loss = q_loss + p_loss

        return loss, p_loss, q_loss

    def calculate_crit_grad(self, y: Tensor, _idx: Optional[Tensor]):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()
        self._noise_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        loss, _, _ = self.crit(y, _idx)
        loss.backward()

    def calculate_crit_grad_p(self, y: Tensor, _idx: Optional[Tensor]):
        # Entry for testing

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        _, p_loss, _ = self.crit(y, _idx)
        p_loss.backward()

    def calculate_crit_grad_q(self, y: Tensor, _idx: Optional[Tensor]):
        # Entry for testing

        # Clear gradients to avoid any issues
        self._noise_distr.clear_gradients()

        # This should automatically assign gradients to model parameters
        _, _, q_loss = self.crit(y, _idx)
        q_loss.backward()

    def calculate_inner_crit_grad(self, y: Tensor, y_samples: Tensor):

        # Clear gradients to avoid any issues
        self._unnorm_distr.clear_gradients()
        self._noise_distr.clear_gradients()

        y_o, y_u, observed_mask = self._mask_input(y)

        q, context = self._noise_distr.forward(y_o, observed_mask)
        y_samples = self.inner_sample_noise(q, num_samples=self._num_neg)

        # This should automatically assign gradients to model parameters
        self.inner_crit((y_o, y_u, observed_mask, context), (y_samples, q)).backward()

    def sample_noise(self, num_samples: int, y: Tensor, q=None):
        with torch.no_grad():
            y_samples = self._noise_distr.sample(torch.Size((num_samples,)), y)

        return y_samples

    def inner_sample_noise(self, q, num_samples: int):
        with torch.no_grad():
            y_samples = self._noise_distr.inner_sample(q, torch.Size((num_samples,)))

        return y_samples

    def part_fn(self, y, y_samples) -> Tensor:
        """Compute Ẑ"""
        pass

    def _mask_input(self, y):
        observed_mask = self.mask_generator(y.shape[0], y.shape[-1])

        return y * observed_mask, y * (1 - observed_mask), observed_mask

    def _generate_model_input(self, y_u, y_samples, context, selected_features=None):

        # TODO: should I not mask y_samples?
        ys_u = concat_samples(y_u, y_samples)

        # TODO: This means select all?
        u_i = torch.broadcast_to(torch.arange(0, y_u.shape[-1], dtype=torch.int64),
            [y_u.shape[0], 1 + self._num_neg, y_u.shape[-1]],
        )

        # TODO: consider selected_features for inference
        if selected_features is not None:
            ys_u = ys_u[:, :, selected_features]# TODO: does this work as I intend?
            u_i = u_i[:, :, selected_features]
            context = context[:, selected_features, :]

        y_u_i = ys_u.reshape(-1)
        u_i = u_i.reshape(-1)
        tiled_context = torch.tile(context.unsqueeze(dim=1), [1, 1 + self._num_neg, 1, 1]).reshape(-1, context.shape[-1])

        return y_u_i, u_i, tiled_context

    def get_proposal(self):
        return self._noise_distr
