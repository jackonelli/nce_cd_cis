# Adapted from https://github.com/conormdurkan/autoregressive-energy-machines
import math
import sys
import argparse
import torch

from numbers import Number
from torch import distributions
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all

from scipy.stats import truncnorm

q = sys.exit


class InfiniteLoader:
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, dataset, batch_size=1, shuffle=True,
                 drop_last=True, num_epochs=None, use_gpu=False):
        """Constructor.
        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """

        if use_gpu:
            device_name = 'cuda'
        else:
            device_name = 'cpu'

        self.loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=torch.Generator(device=device_name)
        )
        self.finite_iterable = iter(self.loader)
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = iter(self.loader)
            return next(self.finite_iterable)

    def __iter__(self):
        return self


class Normal_(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    Example::
         m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
         m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Normal_, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal_, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal_, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_scale = math.log(self.scale) if isinstance(self.scale,
                                                       Number) else self.scale.log()
        return -0.5 * ((value - self.loc) / self.scale) ** 2 - log_scale - 0.5 * math.log(
            2 * math.pi)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf(
            (value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

class MixtureSameFamily(distributions.Distribution):
    def __init__(self, mixture_distribution, components_distribution):
        self.mixture_distribution = mixture_distribution
        self.components_distribution = components_distribution

        super().__init__(
            batch_shape=self.components_distribution.batch_shape,
            event_shape=self.components_distribution.event_shape
        )

    def sample(self, sample_shape=torch.Size()):
        mixture_mask = self.mixture_distribution.sample(sample_shape)  # [S, B, D, M]
        if len(mixture_mask.shape) == 3:
            mixture_mask = mixture_mask[:, None, ...]
        components_samples = self.components_distribution.rsample(
            sample_shape)  # [S, B, D, M]
        samples = torch.sum(mixture_mask * components_samples, dim=-1)  # [S, B, D]
        return samples

    def log_prob(self, value):
        # pad value for evaluation under component density
        value = value.transpose(0, 1) #permute(2, 0, 1)  # [S, B, D]
        value = value[..., None].repeat(1, 1, 1, self.batch_shape[-1])  # [S, B, D, M]
        log_prob_components = self.components_distribution.log_prob(value).transpose(0, 1) #.permute(1, 2, 3, 0)

        # calculate numerically stable log coefficients, and pad
        log_prob_mixture = self.mixture_distribution.logits
        log_prob_mixture = log_prob_mixture.unsqueeze(dim=1)  #[..., None]
        return torch.logsumexp(log_prob_mixture + log_prob_components, dim=-1)


class MixtureSameFamily1D(distributions.Distribution):
    def __init__(self, mixture_distribution, components_distribution):
        self.mixture_distribution = mixture_distribution
        self.components_distribution = components_distribution

        super().__init__(
            batch_shape=self.components_distribution.batch_shape,
            event_shape=self.components_distribution.event_shape
        )

    def sample(self, sample_shape=torch.Size()):
        mixture_mask = self.mixture_distribution.sample(sample_shape)  # [B, 1, M]
        
        if mixture_mask.shape[-1] == 1:
            assert torch.allclose(mixture_mask, torch.ones(mixture_mask.shape))
        

        if len(mixture_mask.shape) == 3:
            mixture_mask = mixture_mask[:, None, ...]
        components_samples = self.components_distribution.rsample(
            sample_shape)  # [S, B, D, M]
        samples = torch.sum(mixture_mask * components_samples, dim=-1).squeeze(dim=0)  # [B, 1]
        return samples

    def log_prob(self, value):
        # pad value for evaluation under component density
        # [B, 1]
        value = value[None, :, :, None].repeat(1, 1, 1, self.batch_shape[-1])  # [S, B, D, M]
        log_prob_components = self.components_distribution.log_prob(value).squeeze(dim=0)  # [B, D, M]

        # calculate numerically stable log coefficients, and pad
        log_prob_mixture = self.mixture_distribution.logits
        return torch.logsumexp(log_prob_mixture + log_prob_components, dim=-1)


def get_aem_losses(data_loader, criterion, device):
    loss, loss_q, loss_p, n = 0, 0, 0, 0

    for y in data_loader:
        y = y.to(device)
        l, l_p, l_q = criterion.crit(y, None)
        loss += l * y.shape[0]
        loss_p += l_p * y.shape[0]
        loss_q += l_q * y.shape[0]
        n += y.shape[0]

    return loss / n, loss_p / n, loss_q / n


def parse_activation(activation):
    activations = {
        'relu': torch.nn.functional.relu,
        'tanh': torch.tanh,
        'sigmoid': torch.nn.functional.sigmoid,
        'softplus': torch.nn.functional.softplus
    }
    return activations[activation]
    
    
def adaptive_resampling(ess, num_samples):
    return ess < (num_samples / 2)
    
def standard_resampling(ess, num_samples):
    # Always resample
    return ess <= num_samples

def parse_args():
    parser = argparse.ArgumentParser()

    # CUDA
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU.')

    # data
    parser.add_argument('--dataset_name', type=str, default='power', help='Dataset to use.')
    parser.add_argument('--train_batch_size', type=int, default=512,
                        help='Size of batch used for training.')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of validation set to use for training/evaluation.')
    parser.add_argument('--val_split', type=str, default='test',
                        help='Which dataset to use for evaluation (train/val/test).')
    parser.add_argument('--val_batch_size', type=int, default=512,
                        help='Size of batch used for validation.')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='Size of batch used for validation.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers used in data loaders.')
    parser.add_argument('--dims', type=int, default=None,
                        help='Number of dimensions to use.')

    # MADE
    parser.add_argument('--n_residual_blocks_made', default=2, type=int,
                        help='Number of residual blocks in MADE.')
    parser.add_argument('--hidden_dim_made', default=256, type=int,
                        help='Dimensionality of hidden layers in MADE.')
    parser.add_argument('--activation_made', default='relu',
                        help='Activation function for MADE.')
    parser.add_argument('--use_batch_norm_made', default=False,
                        help='Whether to use batch norm in MADE.')
    parser.add_argument('--dropout_probability_made', default=0.1, type=float,
                        help='Dropout probability for MADE.')

    # energy net
    parser.add_argument('--context_dim', default=64,
                        help='Dimensionality of context vector.')
    parser.add_argument('--n_residual_blocks_energy_net', default=4,
                        help='Number of residual blocks in energy net.')
    parser.add_argument('--hidden_dim_energy_net', default=128,
                        help='Dimensionality of hidden layers in energy net.')
    parser.add_argument('--energy_upper_bound', default=0, type=float,
                        help='Max value for output of energy net.')
    parser.add_argument('--activation_energy_net', default='relu',
                        help='Activation function for energy net.')
    parser.add_argument('--use_batch_norm_energy_net', default=False,
                        help='Whether to use batch norm in energy net.')
    parser.add_argument('--dropout_probability_energy_net', default=0.1, type=float,
                        help='Dropout probability for energy net.')
    parser.add_argument('--scale_activation', default='softplus',
                        help='Activation to use for scales in proposal mixture components.')
    parser.add_argument('--apply_context_activation', default=False,
                        help='Whether to apply activation to context vector.')

    # proposal
    parser.add_argument('--n_mixture_components', default=10, type=int,
                        help='Number of proposal mixture components (per dimension).')
    parser.add_argument('--proposal_component', default='gaussian',
                        help='Type of location-scale family distribution '
                             'to use in proposal mixture.')
    parser.add_argument('--n_proposal_samples_per_input', default=20,
                        help='Number of proposal samples used to estimate '
                             'normalizing constant during training.')
    parser.add_argument('--n_proposal_samples_per_input_validation', default=20,
                        help='Number of proposal samples used to estimate '
                             'normalizing constant during validation.')
    parser.add_argument("--n_importance_samples", type=int, default=10000,
                        help="Number of importance samples used to estimate norm constant")
    parser.add_argument('--mixture_component_min_scale', default=1e-3,
                        help='Minimum scale for proposal mixture components.')

    # optimization
    parser.add_argument('--learning_rate', default=5e-4,
                        help='Learning rate for Adam.')
    parser.add_argument('--n_total_steps', default=500000, type=int,
                        help='Number of total training steps.')
    parser.add_argument('--alpha_warm_up_steps', default=5000, type=int,
                        help='Number of warm-up steps for aem density.')
    parser.add_argument('--hard_alpha_warm_up', default=True,
                        help='Whether to use a hard warm up for alpha')
    
    # logging and checkpoints
    parser.add_argument('--monitor_interval', default=1000, type=int,
                        help='Interval in steps at which to report training stats.')
    parser.add_argument('--save_interval', default=10000,
                        help='Interval in steps at which to save model.')
    parser.add_argument('--reps', default=1,
                        help='Number of experiment repeats.')
    parser.add_argument('--criterion', default='is',
                        help='Criterion to use (is/cis/pers/adaptive/pers_adaptive).')

    # reproducibility
    parser.add_argument('--seed', default=1638128, type=int,
                        help='Random seed for PyTorch and NumPy.')

    args = parser.parse_args()

    return args

