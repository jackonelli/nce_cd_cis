from pathlib import Path
import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

sys.path.append("..")
from src.nce.cd_cnce import CdCnceCrit
from src.nce.cd_mh_cnce import CdMHCnceCrit
from src.nce.per_cnce import PersistentCondNceCrit
from src.nce.per_cnce_batch import PersistentCondNceCritBatch
from src.nce.per_cnce_batch import PersistentCondNceCritBatch
from src.nce.per_mh_cnce_batch import PersistentMHCnceCritBatch

from src.noise_distr.conditional_normal import ConditionalMultivariateNormal

from src.models.ring_model.ring_model import RingModel, RingModelNCE, unnorm_ring_model_log_pdf
from src.data.ring_model_dataset import RingModelDataset

from src.training.model_training import train_model
from src.training.training_utils import PrecisionErrorMetric, no_stopping, remove_file

from src.utils.ring_model_exp_utils import generate_true_params, initialise_params, get_cnce_noise_distr_par

from src.utils.plot_utils import skip_list_item

D = 5 # Dimension
BATCH_SIZE = 20


def main(args):

    # Create save directories
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    data_dir = Path(str(args.save_dir) + "/datasets")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)


    # Scale learning rate to batch size
    lr = args.base_lr * BATCH_SIZE ** 0.5
    lr_factor = 0.1  # Factor for decaying learning rate

    # Get critera
    configs = get_criteria()
    
    error_res = np.zeros((len(configs), int(np.ceil(args.num_samples / BATCH_SIZE) * args.num_epochs) + 1, args.num_runs))
    acc_prob_res = np.zeros((len(configs), 2, int(np.ceil(args.num_samples / BATCH_SIZE) * args.num_epochs), args.num_runs))

    # Run reps
    for rep in range(args.num_runs):
        print(f"Run {rep + 1}/{args.num_runs}")

        # Get data 
        mu, precision, _ = generate_true_params()
        error_metric = PrecisionErrorMetric(true_precision=precision).metric            

        training_data = RingModelDataset(sample_size=args.num_samples, num_dims=D, mu=mu, precision=precision, 
                                         root_dir=os.path.join(data_dir, "ring_data_size_" + str(args.num_samples) + "_nn_" + str(args.num_neg_samples) + "_rep_" + str(rep)))
        train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

        # Initialise           
        _, log_precision_init, log_z_init = initialise_params()

        # Get noise distr. params
        # p_m = RingModel(mu=mu, log_precision=log_precision_init.clone())
        cov_noise_cnce = get_cnce_noise_distr_par(training_data.get_full_data()) 

        for i, config in enumerate(configs):

            # Remove old acc. prob.
            remove_file(os.path.join(args.save_dir, config["name"] + "_num_neg_" + str(args.num_neg_samples) + "_cd_cnce_acc_prob.npy"))
            remove_file(os.path.join(args.save_dir, config["name"] + "_num_neg_" + str(args.num_neg_samples) + "_cd_mh_acc_prob.npy"))

            # Make sure that these are "reinitialised"
            p_m, p_n, criterion = None, None, None

            p_m = RingModel(mu=mu, log_precision=log_precision_init.clone())
            p_n = ConditionalMultivariateNormal(cov=cov_noise_cnce)
            
            if config["mcmc_steps"] is not None: 
                criterion = config["criterion"](p_m, p_n, args.num_neg_samples, config["mcmc_steps"], save_acc_prob=config["calc_acc_prob"], save_dir=args.save_dir)
            else:
                criterion = config["criterion"](p_m, p_n, args.num_neg_samples, save_acc_prob=config["calc_acc_prob"], save_dir=args.save_dir)

            _, error_res[i, :, rep] = train_model(criterion, error_metric, train_loader, num_epochs=args.num_epochs,
                               stopping_condition=no_stopping, lr=lr, decaying_lr=True, lr_factor=lr_factor)
            # TODO: extract metrices here instead of

            # Fetch data that has been saved (acceptance prob saved in criteria)
            save_dir_pre = os.path.join(args.save_dir, config["name"] + "_num_neg_" + str(args.num_neg_samples))
            if config["calc_acc_prob"]:
                acc_prob_res[i, 0, :, rep] = np.load(Path(str(save_dir_pre) + "_" + "cd_cnce" + "_acc_prob.npy")).mean(axis=0)
                acc_prob_res[i, 1, :, rep] = np.load(Path(str(save_dir_pre) + "_" + "cd_mh" + "_acc_prob.npy")).mean(axis=0)

    plot_acc_prob_res(configs, error_res, acc_prob_res, args)


def plot_acc_prob_res(configs, error_res, acc_prob_res, args):
    # Visualise result
    num_its = error_res.shape[-2] - 1
    x = skip_list_item(np.arange(1, num_its), args.data_res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].set_yscale('log')


    for i, config in enumerate(configs):

        err_data = error_res[i, x, :]
        cnce_data = acc_prob_res[i, 0, x, :]
        mh_cnce_data = acc_prob_res[i, 1, x, :]

        err_median, err_worst_case = np.median(err_data, axis=-1), np.max(err_data, axis=-1)

        ax[0].plot(x, err_median, color=config["color"], label=config["label"])
        ax[0].plot(x, err_worst_case, '--', color=config["color"])

        cnce_median = np.median(cnce_data, axis=-1)
        mh_cnce_median = np.median(mh_cnce_data, axis=-1)

        if config["label"] in ['CNCE', 'P-CNCE']:
            other_col, other_lab = ([124/255,161/255,204/255], 'MH-CNCE') if config["label"]=='CNCE' else ([238/255,186/255,180/255], 'P-MH-CNCE')
            ax[1].plot(x, cnce_median, color=config["color"], label=config["label"])
            ax[1].plot(x, mh_cnce_median, color=other_col, label=other_lab)
        elif config["label"] in ['MH-CNCE', 'P-MH-CNCE']:
            other_col, other_lab = ([31/255,68/255,156/255], 'CNCE') if config["label"]=='MH-CNCE' else ([240/255,80/255,57/255], 'P-CNCE')
            ax[2].plot(x, cnce_median, color=other_col, label=other_lab)
            ax[2].plot(x, mh_cnce_median, color=config["color"], label=config["label"])


        ax[0].set_xlabel("Iter.")
        ax[0].set_ylabel("Sq. Error")

        ax[1].set_xlabel("Iter.")
        ax[1].set_ylabel("Acc. Prob., (P-)CNCE")
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, fancybox=True, shadow=True)

        ax[2].set_xlabel("Iter.")
        ax[2].set_ylabel("Acc. Prob., (P-)MH-CNCE")

        tikzplotlib.save(os.path.join(args.save_dir, "cnce_acc_prob_res_num_samples_" + str(args.num_samples) + "_num_neg_samples_" + str(args.num_neg_samples) + ".tex"))
        plt.savefig(os.path.join(args.save_dir, "cnce_acc_prob_res_num_samples_" + str(args.num_samples) + "_num_neg_samples_" + str(args.num_neg_samples) + ".png"))

    plt.show()


def get_criteria():

    config_cnce = {
        "criterion": CdCnceCrit,
        "name": "cd_cnce",
        "conditional_noise_distr": True,
        "mcmc_steps": 1,
        "calc_acc_prob": True,
        "label": 'CNCE',
        "color": [31/255,68/255,156/255]
    }

    config_pers_cnce = {
        "criterion": PersistentCondNceCritBatch,
        "name": "pers_cnce",
        "conditional_noise_distr": True,
        "mcmc_steps": None,
        "calc_acc_prob": True,
        "label": 'P-CNCE',
        "color": [240/255,80/255,57/255]
    }
    
    config_mh_cnce = {
        "criterion": CdMHCnceCrit,
        "name": "cd_mh",
        "conditional_noise_distr": True,
        "mcmc_steps": 1,
        "calc_acc_prob": True,
        "label": 'MH-CNCE',
        "color": [124/255,161/255,204/255]
    }

    config_pers_mh_cnce = {
        "criterion": PersistentMHCnceCritBatch,
        "name": "pers_mh_cnce",
        "conditional_noise_distr": True,
        "mcmc_steps": None,
        "calc_acc_prob": True,
        "label": 'P-MH-CNCE',
        "color": [238/255,186/255,180/255]
    }

    return [config_cnce, config_pers_cnce, config_mh_cnce, config_pers_mh_cnce]
        
    
def parse_args():
    parser = argparse.ArgumentParser(
        "Ring model experiments with (pers.) (MH-)CNCE."
    )
    parser.add_argument(
        "--num-epochs", default=50, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--num-runs", default=100, type=int, help="Number of runs to average over."
    )
    parser.add_argument(
        "--num-samples", default=200, type=int, help="Number of training samples."
    )
    parser.add_argument(
        "--num-neg-samples", default=5, type=int, help="Number of negative examples used in criteria."
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
        default="experiments/res/acc_prob/",
        type=Path,
        help="Directory where results should be saved.",
    )
    return parser.parse_args()



if __name__ == "__main__":
    main(parse_args())