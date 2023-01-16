import torch
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def generate_tikz_data_table(file_path, data, header):
    assert len(header) == data.size(1), "Num columns and labels mismatch."
    str_ = [",".join(header) + "\n"]
    for row in data:
        str_.append(",".join(map(str, row.numpy())) + "\n")

    with open(file_path, "w") as f:
        f.writelines(str_)


def mvn_curve(mu, cov, std=1, res=100):
    with torch.no_grad():
        angles = torch.linspace(0, 2 * torch.pi, res)
        curve_param = torch.column_stack((torch.cos(angles), torch.sin(angles)))
        ellipsis = std * curve_param @ torch.Tensor(sqrtm(cov))
        return mu + ellipsis


def plot_mvn(levels, ax, label):
    ax.plot(levels[:, 0], levels[:, 1], label=label)


def plot_distrs_ideal(p_d, p_t_d, p_t_t):
    _, ax = plt.subplots()
    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    distrs = [
        (p_d.mu, p_d.cov, "$p_{d}}$"),
        (p_t_d.mu, p_t_d.cov(), "$q=p_d$"),
        (p_t_t.mu, p_t_t.cov(), "$q = p_{\\theta}$"),
    ]
    for mu, cov, label in distrs:
        plot_mvn(mvn_curve(mu, cov), ax, label)
    ax.set_title("Comparison, optimal proposal distrs.")
    ax.legend()


def plot_distrs_adaptive(p_d, p_theta, q_phi):
    _, ax = plt.subplots()
    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    distrs = [
        (p_d.mu, p_d.cov, "$p_{d}}$"),
        (p_theta.mu, p_theta.cov(), "$p_{\\theta}$"),
        (q_phi.mu, q_phi.cov(), "$q_{\\varphi}$"),
    ]
    for mu, cov, label in distrs:
        plot_mvn(mvn_curve(mu, cov), ax, label)
    ax.set_title("Adaptive proposal")
    ax.legend()
