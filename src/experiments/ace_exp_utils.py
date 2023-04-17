# Adapted from https://github.com/lupalab/ace/blob/main/ace/masking.py
from abc import ABC, abstractmethod
import argparse
import torch


class MaskGenerator(ABC):
    def __init__(
        self,
        seed=None,
        device=torch.device("cpu")
    ):
        if seed is not None:
            self.gen = torch.Generator().manual_seed(seed)    #device="cuda"
        else:
            self.gen = None

        self.device = device

    def __call__(self, num_samples, num_features):
        return self.call(num_samples, num_features).to(self.device)

    @abstractmethod
    def call(self, num_samples, num_features):
        pass


class UniformMaskGenerator(MaskGenerator):
    def call(self, num_samples, num_features):

        # For each obs., observe 0 to num_features-1 features
        k = torch.randint(low=0, high=num_features, size=(num_samples,), generator=self.gen)

        result = []
        for i in range(num_samples):
            mask = torch.zeros(num_features)
            inds = torch.randperm(num_features, generator=self.gen)[:k[i]]
            mask[inds] = 1
            result.append(mask)

        return torch.vstack(result)


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.5, seed=None, device=torch.device("cpu")):
        super().__init__(seed, device)

        self.p = p

    def call(self, num_samples, num_features):
        return torch.bernoulli(torch.tensor([self.p] * num_samples * num_features),
                               generator=self.gen).reshape(num_samples, num_features)


def get_ace_losses(data_loader, criterion, device):
    loss, loss_q, loss_p = 0, 0, 0

    for (y, idx) in data_loader:
        y = y.to(device)
        l, l_p, l_q = criterion.crit(y, None)
        loss += l * y.shape[0]
        loss_p += l_p * y.shape[0]
        loss_q += l_q * y.shape[0]

    return loss / len(data_loader.dataset), loss_p / len(data_loader.dataset), loss_q / len(data_loader.dataset)

def parse_args():
    """Arg parser"""

    parser = argparse.ArgumentParser(description="ACE experiments")
    parser.add_argument("--noise_scale", type=int, default=0.001, help="Scale of noise added to train data")
    parser.add_argument("--proposal_num_blocks", type=int, default=4, help="Number of Res. BLocks in proposal")
    parser.add_argument("--proposal_num_hidden", type=int, default=512, help="Number of hidden units in proposal")
    parser.add_argument("--num_context_units", type=int, default=64, help="Dimension of latent representation")
    parser.add_argument("--num_negative", type=int, default=20, help="Number of negative samples for loss evaluation")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Loss trade-off parameter")
    parser.add_argument("--energy_reg", type=float, default=0.0, help="Loss regularisation weight")
    parser.add_argument("--num_warm_up_steps", type=int, default=5000, help="Number of warmup steps")
    parser.add_argument("--num_training_steps", type=int, default=1000000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr_factor", type=float, default=0.0001, help="Minimal learning rate will be lr_factor*lr")
    parser.add_argument("--num_steps_decay", type=int, default=1000000,
                        help="Number of training steps to decrease learning rate")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Number of training steps to decrease learning rate")
    parser.add_argument("--num_is_samples", type=int, default=1000, help="Number of IS samples for log. likelihood evaluation")
    parser.add_argument("--num_permutations", type=int, default=1, help="Number of permutations for log. likelihood evaluation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed, NB both cuda and cpu")  # Note: not used
    parser.add_argument("--save_dir", type=str, default="./res", help="Save directory")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--reps", type=int, default=1, help="Number of times to repeat experiment")

    return parser.parse_args()