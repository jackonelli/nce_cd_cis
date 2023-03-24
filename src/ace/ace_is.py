"""Importance Sampling (IS) for ACE"""
from typing import Optional
import torch
from torch import Tensor

from src.part_fn_base import PartFnEstimator
from src.part_fn_utils import concat_samples

from src.noise_distr.ace_proposal import AceProposal
from src.models.ace.ace_model import AceModel
from src.experiments.ace_exp_utils import UniformMaskGenerator, BernoulliMaskGenerator


class AceIsCrit(PartFnEstimator):
    def __init__(
        self,
        unnorm_distr: AceModel,
        noise_distr: AceProposal,
        num_neg_samples: int,
        alpha: float = 1.0,
        energy_reg: float = 0.0,
        mask_generator=None,
        device=torch.device("cpu"),
        batch_size=None
    ):
        super().__init__(unnorm_distr, noise_distr, num_neg_samples)

        self.mcmc_steps = 1  # For now, this option is not available
        self.alpha = alpha  # For regularisation
        self.energy_reg = energy_reg  # TODO: In other code, they assign this to the data
        self.device = device
        if mask_generator is None:
            self.mask_generator = UniformMaskGenerator(device=self.device)
        else:
            self.mask_generator = mask_generator  # TODO: set seed?

    def crit(self, y: Tensor, _idx: Optional[Tensor]):
        # Mask input
        y_o, y_u, observed_mask = self._mask_input(y)

        q, context = self._noise_distr.forward((y_o, observed_mask))
        y_samples = self.inner_sample_noise(q, num_samples=self._num_neg)

        return self.inner_crit((y_u, observed_mask, context), (y_samples, q))

    def inner_crit(self, y: tuple, y_samples: tuple, y_base=None):

        # Note that we calculate the criterion and not the gradient directly
        # Note: y, y_samples are tuples

        y_u, observed_mask, context = y
        y_samples, q = y_samples

        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples, _ = \
            self._log_probs(y_u, y_samples, observed_mask, context, q)

        assert log_p_tilde_y.shape == (y_u.shape[0], y_u.shape[-1])
        assert log_q_y.shape == (y_u.shape[0], y_u.shape[-1])

        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples.detach().clone()) * (1 - observed_mask).unsqueeze(dim=1)
        log_z = (torch.logsumexp(log_w_tilde_y_samples, dim=1) - torch.log(torch.tensor(self._num_neg))) * (
                    1 - observed_mask)

        log_p_y = log_p_tilde_y - log_z
        assert log_p_y.shape == (y_u.shape[0], y_u.shape[-1])

        #is_weights = torch.nn.Softmax(dim=1)(log_w_tilde_y_samples)
        #energy_mean = torch.sum(is_weights * y_samples_u) * (1 - observed_mask)

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
        loss, lp, lq = self.crit(y, _idx)
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
        # # Clear gradients to avoid any issues
        # self._unnorm_distr.clear_gradients()
        # self._noise_distr.clear_gradients()
        #
        # y_o, y_u, observed_mask = self._mask_input(y)
        #
        # q, context = self._noise_distr.forward(y_o, observed_mask)
        # y_samples = self.inner_sample_noise(q, num_samples=self._num_neg)
        #
        # # This should automatically assign gradients to model parameters
        # self.inner_crit((y_o, y_u, observed_mask, context), (y_samples, q)).backward()

        pass

    def sample_noise(self, num_samples: int, y: Tensor, q=None):
        with torch.no_grad():
            y_samples = self._noise_distr.sample(torch.Size((num_samples,)), y)

        return y_samples

    def inner_sample_noise(self, q, num_samples: int):
        with torch.no_grad():
            y_samples = self._noise_distr.inner_sample(q, torch.Size((num_samples,)))

        return y_samples

    def part_fn(self, y, num_samples=1000) -> Tensor:
        """Compute Ẑ"""
        pass

    def log_likelihood(self, y, num_samples=20, num_permutations=1):
        # Likelihood estimates are computed with 20,000 importance samples for POWER, GAS, and HEPMASS,
        # 10,000 importance samples for MINIBOONE, and 3,000 importance samples for BSDS.
        # Results are averaged over 5 observed masks.

        self._unnorm_distr.eval()
        self._noise_distr.eval()

        with torch.no_grad():

            if num_permutations == 1:
                y_o, y_u, observed_mask = self._mask_input(y, mask=BernoulliMaskGenerator(device=self.device))
                return self.single_permutation_ll(y, observed_mask, num_samples)

            else:
                ll = torch.zeros((num_permutations,)).to(self.device)
                for i in range(num_permutations):
                    y_o, y_u, observed_mask = self._mask_input(y, mask=BernoulliMaskGenerator(device=self.device))
                    ll[i] = self.single_permutation_ll(y, observed_mask, num_samples)

                return ll

    def single_permutation_ll(self, y, observed_mask, num_samples=20):

        # Create random permutations of query
        expanded_y, expanded_observed_mask, selected_features = [], [], []
        for i in range(y.shape[0]):
            if torch.count_nonzero(1 - observed_mask[i, :]) > 0:
                y_e, mask, sf = self._permute_mask(y[i, :], observed_mask[i, :])
                expanded_y.append(y_e)
                expanded_observed_mask.append(mask)
                selected_features.append(sf)

        expanded_y, expanded_observed_mask, selected_features = torch.cat(expanded_y, dim=0), \
                                                                torch.cat(expanded_observed_mask, dim=0), \
                                                                torch.cat(selected_features, dim=0)


        # Calculate log likelihood
        expanded_y_o, expanded_y_u = expanded_y * expanded_observed_mask, expanded_y * (1 - expanded_observed_mask)
        q, context = self._noise_distr.forward((expanded_y_o, expanded_observed_mask))
        y_samples = self.inner_sample_noise(q, num_samples=num_samples)

        # Calculate log prob for model
        log_p_tilde_y, log_p_tilde_y_samples, log_q_y, log_q_y_samples, new_observed_mask \
            = self._log_probs(expanded_y_u, y_samples, expanded_observed_mask, context, q, selected_features)

        # Estimate z with importance sampling
        log_w_tilde_y_samples = (log_p_tilde_y_samples - log_q_y_samples) \
                                * (1 - new_observed_mask).unsqueeze(dim=1)

        log_z = (torch.logsumexp(log_w_tilde_y_samples, dim=1) - torch.log(torch.tensor(num_samples))) * (
                1 - new_observed_mask)

        return torch.sum(log_p_tilde_y - log_z) / y.shape[0]

    def _permute_mask(self, y, observed_mask):
        query = 1 - observed_mask
        u_inds = torch.nonzero(query).squeeze(dim=1)
        u_inds = u_inds[torch.randperm(u_inds.shape[0])]

        # "Exclusive" cumsum
        expanded_mask = torch.cumsum(torch.nn.functional.one_hot(u_inds, y.shape[-1]).float(), dim=0)
        expanded_mask[-1, :] = 0
        expanded_mask = expanded_mask.roll(shifts=1, dims=0)

        expanded_mask += observed_mask.unsqueeze(dim=0)
        y = torch.tile(y.unsqueeze(dim=0), [u_inds.shape[0], 1])

        return y, expanded_mask, u_inds.unsqueeze(dim=1)

    def _log_probs(self, y_u, y_samples, observed_mask, context, q, selected_features=None):
        """Helper function for calculating p_y, p_y_samples, q_y, q_y_samples"""

        # Mask samples
        y_samples_u = y_samples * (1 - observed_mask).unsqueeze(dim=1)
        assert y_samples_u.shape == y_samples.shape

        # Calculate log prob for proposal
        log_q_y = self._noise_distr.inner_log_prob(q, y_u) * (1 - observed_mask)
        log_q_y_samples = self._noise_distr.inner_log_prob(q, y_samples.transpose(0, 1)).transpose(0, 1)
        log_q_y_samples = log_q_y_samples * (1 - observed_mask).unsqueeze(dim=1)

        assert log_q_y_samples.shape == y_samples.shape

        # q_mean = q.mean * (1 - observed_mask)

        # Calculate log prob for model
        # TODO: when selected_features is None; isn't it unnecessary still that we do predictions for all features (as we mask the observed later on)
        y_u_i, u_i, tiled_context = self._generate_model_input(y_u, y_samples_u, context, selected_features)
        log_p_tilde_ys = self._unnorm_distr.log_prob((y_u_i, u_i, tiled_context))\

        if selected_features is None:
            log_p_tilde_ys = log_p_tilde_ys.reshape(-1, y_samples.shape[1] + 1, y_u.shape[-1])
        else:
            assert selected_features.ndim == 2
            log_p_tilde_ys = log_p_tilde_ys.reshape(-1, y_samples.shape[1] + 1, selected_features.shape[-1])

            observed_mask = torch.gather(observed_mask, dim=1, index=selected_features)
            log_q_y = torch.gather(log_q_y, dim=1, index=selected_features)
            log_q_y_samples = torch.gather(log_q_y_samples, dim=-1,
                                           index=selected_features.repeat(1, log_q_y_samples.shape[1]).unsqueeze(dim=-1))
            #y_samples = torch.gather(y_samples, dim=-1,
                                     #index=selected_features.repeat(1, y_samples.shape[1]).unsqueeze(dim=-1))

        log_p_tilde_ys *= (1 - observed_mask).unsqueeze(dim=1)
        assert log_p_tilde_ys.shape[0] == y_u.shape[0]

        log_p_tilde_y, log_p_tilde_y_samples = log_p_tilde_ys[:, 0, :], log_p_tilde_ys[:, 1:, :]  # TODO: not last col?

        return log_p_tilde_y, log_p_tilde_y_samples, log_q_y.type(torch.float32), log_q_y_samples.type(torch.float32), observed_mask

    def _mask_input(self, y, mask=None):
        if mask is None:
            observed_mask = self.mask_generator(y.shape[0], y.shape[-1])
        else:
            observed_mask = mask(y.shape[0], y.shape[-1])

        return y * observed_mask, y * (1 - observed_mask), observed_mask

    def _generate_model_input(self, y_u, y_samples_u, context, selected_features=None):

        # TODO: should I not mask y_samples?
        ys_u = concat_samples(y_u, y_samples_u)

        u_i = torch.broadcast_to(torch.arange(0, y_u.shape[-1], dtype=torch.int64, device=self.device),
                                 [y_u.shape[0], 1 + y_samples_u.shape[1], y_u.shape[-1]],
                                 )

        if selected_features is not None:
            ys_u = torch.gather(ys_u, dim=-1, index=selected_features.repeat(1, ys_u.shape[1]).unsqueeze(dim=-1))
            u_i = torch.gather(u_i, dim=-1, index=selected_features.repeat(1, u_i.shape[1]).unsqueeze(dim=-1))
            context = torch.gather(context, dim=1, index=selected_features.repeat(1, context.shape[-1]).unsqueeze(dim=1))

        y_u_i = ys_u.reshape(-1)
        u_i = u_i.reshape(-1)
        tiled_context = torch.tile(context.unsqueeze(dim=1), [1, 1 + y_samples_u.shape[1], 1, 1]).reshape(-1, context.shape[-1])

        return y_u_i, u_i, tiled_context

    def get_proposal(self):
        return self._noise_distr
