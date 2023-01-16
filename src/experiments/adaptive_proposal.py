"""This code generates the figure 1"""

from pathlib import Path
import argparse
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append("..")
from src.nce.rank import NceRankCrit

from src.noise_distr.normal import MultivariateNormal
from src.models.gaussian_model import DiagGaussianModel
from src.nce.adaptive_rank import AdaptiveRankKernel
from src.noise_distr.adaptive import AdaptiveDiagGaussianModel

from src.training.model_training import (
    train_model,
    train_model_model_proposal,
    train_model_adaptive_proposal,
)
from src.data.normal import MultivariateNormalData
from src.training.training_utils import (
    MvnKlDiv,
    no_change_stopping_condition,
)
from src.experiments.utils import generate_tikz_data_table


def main(args):
    D, N, J = 5, 100, 10  # Dimension, Num. data samples, Num neg. samples
    mu_star, cov_star = (
        torch.ones(
            D,
        ),
        torch.eye(D),
    )

    # Data distribution
    p_d = MultivariateNormal(mu_star, cov_star)
    # Model distribution
    init_mu, init_cov = (
        5.0
        * torch.ones(
            D,
        ),
        4.0 * torch.eye(D),
    )

    # Optimisation
    num_epochs = args.num_epochs
    batch_size = N
    learn_rate = args.base_lr * batch_size ** 0.5

    # Metrics
    kl_div = MvnKlDiv(p_d.mu, p_d.cov).metric
    metric = kl_div

    # q = p_d
    p_t_data_noise = DiagGaussianModel(init_mu.clone(), init_cov.clone())
    print("Training with q = p_d")

    training_data = MultivariateNormalData(mu_star, cov_star, N)
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    _, p_d_metrics = train_model(
        NceRankCrit(p_t_data_noise, p_d, J),
        metric,
        train_loader,
        None,
        neg_sample_size=J,
        num_epochs=num_epochs,
        stopping_condition=no_change_stopping_condition,
        lr=learn_rate,
    )
    # q = p_theta
    p_t_model_noise = DiagGaussianModel(init_mu.clone(), init_cov.clone())
    print("Training with q = p_theta")

    _, p_t_metrics = train_model_model_proposal(
        p_t_model_noise,
        NceRankCrit,
        metric,
        train_loader,
        None,
        J,
        num_epochs,
        lr=learn_rate,
        stopping_condition=no_change_stopping_condition,
    )

    # Adaptive model
    p_theta = DiagGaussianModel(init_mu.clone(), init_cov.clone())
    q_phi = AdaptiveDiagGaussianModel(mu_star.clone(), cov_star.clone())
    p_crit = NceRankCrit(p_theta, q_phi, J)
    q_crit = AdaptiveRankKernel(p_theta, q_phi, J)

    _, q_phi_metrics = train_model_adaptive_proposal(
        p_theta,
        q_phi,
        p_crit,
        q_crit,
        metric,
        train_loader,
        None,
        neg_sample_size=J,
        num_epochs=num_epochs,
        stopping_condition=no_change_stopping_condition,
        lr=learn_rate,
    )

    plot_kl_div(p_d_metrics, p_t_metrics, q_phi_metrics)
    if args.save_dir:
        data_file_path = args.save_dir / "klData.dat"
        assert args.save_dir.exists(), f"Save dir. '{args.save_dir}' does not exist."
        print(f"Saving KL metrics to '{data_file_path}'")
        iters = torch.arange(start=0, end=q_phi_metrics.size(0), step=args.data_res)
        data = torch.column_stack(
            (iters, p_d_metrics[iters], p_t_metrics[iters], q_phi_metrics[iters])
        )
        generate_tikz_data_table(
            data_file_path, data, ["t", "pdata", "ptheta", "adaptive"]
        )


def plot_kl_div(p_d_metrics, p_t_metrics, q_phi_metrics):
    _, ax = plt.subplots()
    ax.plot(torch.arange(p_d_metrics.size(0)), p_d_metrics, "-r", label="$q=p_d$")
    ax.plot(
        torch.arange(p_t_metrics.size(0)), p_t_metrics, "-b", label="$q=p_{\\theta}$"
    )
    ax.plot(
        torch.arange(q_phi_metrics.size(0)),
        q_phi_metrics,
        "g--",
        label="$q=q_{\\varphi}$",
    )
    ax.legend()
    # ax.set_xlim([0, 250])
    ax.set_title("Choice of proposal distribution")
    ax.set_xlabel("Iter. step $t$")
    ax.set_ylabel("KL$(p_d || p_{\\theta})$")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        "Toy example with different proposal distributions."
    )
    parser.add_argument(
        "--num-epochs", default=250, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--base-lr",
        default=0.01,
        type=float,
        help="Base learning rate (scales with batch size).",
    )
    parser.add_argument(
        "--data-res",
        default=1,
        type=int,
        help="Data resolution, sets step between data point, to reduce storage.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="Number of epochs to train for.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
