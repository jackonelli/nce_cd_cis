"""This code generates the data for the
"Adaptive proposal distribution" toy example
"""

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
    no_stopping,
)
from src.experiments.utils import generate_bounds, table_data, format_table

D, N, J = 5, 128, 10  # Dimension, Num. data samples, Num neg. samples
# Initial values for p_theta (model distribution)
INIT_MU = 4.0 * torch.ones(
    D,
)
INIT_COV = 4.0 * torch.eye(D)

MU_STAR = torch.zeros(
    D,
)
COV_STAR = torch.eye(D)
BATCH_SIZE = 32


def main(args):
    if args.save_dir is not None:
        assert args.save_dir.exists(), f"Save dir. '{args.save_dir}' does not exist."

    # Optimisation
    num_epochs = args.num_epochs
    learn_rate = args.base_lr * BATCH_SIZE ** 0.5
    # Options for decaying learning rate.
    scheduler_opts = (num_epochs * (N // BATCH_SIZE), 0.01)

    # Metrics
    metric = MvnKlDiv(MU_STAR, COV_STAR).metric

    training_data = MultivariateNormalData(MU_STAR, COV_STAR, N)
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True
    )

    p_d_metrics = torch.empty((args.num_runs, (N // BATCH_SIZE * args.num_epochs) + 1))
    p_t_metrics = torch.empty(p_d_metrics.size())
    q_phi_metrics = torch.empty(p_d_metrics.size())
    for m in range(args.num_runs):
        print(f"Run {m+1}/{args.num_runs}")
        p_d, p_t_data_noise, p_t_model_noise = create_distr()
        # q = q_phi
        p_theta = DiagGaussianModel(INIT_MU.clone(), INIT_COV.clone())
        # Note: we initialise q = p_d to make it a fair comparison.
        q_phi = AdaptiveDiagGaussianModel(MU_STAR.clone(), COV_STAR.clone())
        p_crit = NceRankCrit(p_theta, q_phi, J)
        q_crit = AdaptiveRankKernel(p_theta, q_phi, J)

        _, q_phi_metrics[m, :] = train_model_adaptive_proposal(
            p_theta,
            q_phi,
            p_crit,
            q_crit,
            metric,
            train_loader,
            None,
            neg_sample_size=J,
            num_epochs=num_epochs,
            stopping_condition=no_stopping,
            lr=learn_rate,
            scheduler_opts=scheduler_opts,
        )
        # q = p_d
        _, p_d_metrics[m, :] = train_model(
            NceRankCrit(p_t_data_noise, p_d, J),
            metric,
            train_loader,
            None,
            neg_sample_size=J,
            num_epochs=num_epochs,
            stopping_condition=no_stopping,
            lr=learn_rate,
            scheduler_opts=scheduler_opts,
        )

        # q = p_theta
        _, p_t_metrics[m, :] = train_model_model_proposal(
            p_t_model_noise,
            NceRankCrit,
            metric,
            train_loader,
            None,
            J,
            num_epochs,
            lr=learn_rate,
            stopping_condition=no_stopping,
            scheduler_opts=scheduler_opts,
        )

        # q = q_phi
        p_theta = DiagGaussianModel(INIT_MU.clone(), INIT_COV.clone())
        # Note: we initialise q = p_d to make it a fair comparison.
        q_phi = AdaptiveDiagGaussianModel(MU_STAR.clone(), COV_STAR.clone())
        p_crit = NceRankCrit(p_theta, q_phi, J)
        q_crit = AdaptiveRankKernel(p_theta, q_phi, J)

        _, q_phi_metrics[m, :] = train_model_adaptive_proposal(
            p_theta,
            q_phi,
            p_crit,
            q_crit,
            metric,
            train_loader,
            None,
            neg_sample_size=J,
            num_epochs=num_epochs,
            stopping_condition=no_stopping,
            lr=learn_rate,
            scheduler_opts=scheduler_opts,
        )
    if args.save_dir:
        # Save torch tensors:
        print(f"Saving KL metrics in '{args.save_dir}'")
        for data, name in [
            (p_d_metrics, "p_d"),
            (p_t_metrics, "p_t"),
            (q_phi_metrics, "q_f"),
        ]:
            torch.save(data, args.save_dir / f"{args.num_runs}_runs_kl_raw_{name}.pth")
            with open(args.save_dir / f"{args.num_runs}_runs_{name}.txt", "w") as f:
                f.writelines(
                    format_table(
                        *table_data(*generate_bounds(data)), ["t", "kl", "low", "upp"]
                    )
                )

    plot_kl_div(p_d_metrics, p_t_metrics, q_phi_metrics)


def create_distr():

    # Data distribution
    p_d = MultivariateNormal(MU_STAR, COV_STAR)
    # Model distribution
    p_t_data_noise = DiagGaussianModel(INIT_MU.clone(), INIT_COV.clone())
    p_t_model_noise = DiagGaussianModel(INIT_MU.clone(), INIT_COV.clone())
    return p_d, p_t_data_noise, p_t_model_noise


def plot_kl_div(p_d_metrics, p_t_metrics, q_phi_metrics):
    _, ax = plt.subplots()
    it, med, low, upp = table_data(*generate_bounds(p_d_metrics.numpy()))
    ax.errorbar(it, med, [low, upp], label="$q=p_d$")

    it, med, low, upp = table_data(*generate_bounds(p_t_metrics.numpy()))
    ax.errorbar(it, med, [low, upp], label="$q=p_t$")

    it, med, low, upp = table_data(*generate_bounds(q_phi_metrics.numpy()))
    ax.errorbar(it, med, [low, upp], label="$q=p_{\\theta}$")

    ax.legend()
    ax.set_xlim([1, it.max() + 1])
    ax.set_title("Choice of proposal distribution")
    ax.set_xlabel("Iter. step $t$")
    ax.set_ylabel("KL$(p_d || p_{\\theta})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        "Toy example with different proposal distributions."
    )
    parser.add_argument(
        "--num-epochs", default=2000, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--num-runs", default=20, type=int, help="Number of runs to average over."
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
