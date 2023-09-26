import json
import os
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.aem.aem_cis_joint_z_adapt import AemCisJointAdaCrit
from src.aem.aem_pers_cis import AemCisJointPersCrit
from src.aem.aem_pers_cis_adapt import AemCisJointAdaPersCrit
from src.aem.aem_smc import AemSmcCrit
from src.aem.aem_smc_cond import AemSmcCondCrit
from src.aem.aem_pers_cond_smc import AemSmcCondPersCrit
from src.aem.aem_smc_adaptive import AemSmcAdaCrit
from src.aem.aem_smc_cond_adaptive import AemSmcCondAdaCrit
from src.data import data_uci
from src.data.data_uci.uciutils import get_project_root
from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal
from src.experiments.aem_exp_utils import parse_activation, parse_args, InfiniteLoader

from src.training.model_training import train_aem_model


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_name = args.dataset_name
    proj_dir = os.path.join(get_project_root(), "deep_ext_obj/nbs/res/aem/")
    base_dir = os.path.join(proj_dir, args.dataset_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # (io.get_checkpoint_root())

    # Save args
    #with open(os.path.join(base_dir, 'commandline_args.txt'), 'w') as f:
    #   json.dump(args.__dict__, f, indent=2)

    crit_dir = {
        'is': ([AemIsJointCrit], ["aem_is_j"]),
        'cis': ([AemCisJointCrit], ["aem_cis_j"]),
        'pers': ([AemCisJointPersCrit], ["aem_cis_pers_j"]),
        'adaptive': ([AemCisJointAdaCrit], ["aem_cis_adapt_j"]),
        'pers_adaptive': ([AemCisJointAdaPersCrit], ["aem_cis_pers_adapt_j"]),
        'smc': ([AemSmcCrit], ["aem_smc_j"]),
        'csmc': ([AemSmcCondCrit], ["aem_csmc_j"]),
        'csmc_pers': ([AemSmcCondPersCrit], ["aem_csmc_pers_j"]),
        'smc_adaptive': ([AemSmcAdaCrit], ["aem_smc_adapt_j"]),
        'csmc_adaptive': ([AemSmcCondAdaCrit], ["aem_csmc_adapt_j"]),
    }

    if args.criterion in crit_dir:
        crits, crit_lab = crit_dir[args.criterion]
        
        if args.dims is not None:
            crit_lab = [crit_lab[0] + "_d_" + str(args.dims)]
            
        if args.energy_upper_bound > 0.0:
            crit_lab = [crit_lab[0] + "_ub_" + str(args.energy_upper_bound)]
        
    else:
        crits, crit_lab = [], []
        print("Unknown criterion!")

    ll, ll_std = np.zeros((args.reps, len(crits))), np.zeros((args.reps, len(crits)))
    for i in range(args.reps):
        test_loader = load_data(data_name, args)

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            ll[i, j], ll_std[i, j] = run_test(test_loader, crit, save_dir, args)
            print("Test log. likelihood, mean: {}".format(ll[i, j]))
            print("Test log. likelihood, std: {}".format(ll_std[i, j]))


def load_data(name, args):
    # create datasets

    gen = None

    dataset = data_uci.load_uci_dataset(args.dataset_name, split='test', num_dims=args.dims) #, frac=args.val_frac)
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        generator=gen
    )

    return test_loader


def run_test(test_loader, criterion, save_dir, args):
    # Create energy net

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    dim = test_loader.dataset.dim  # D
    output_dim_multiplier = args.context_dim + 3 * args.n_mixture_components  # K + 3M

    model = ResidualEnergyNet(
        input_dim=(args.context_dim + 1),
        n_residual_blocks=args.n_residual_blocks_energy_net,
        hidden_dim=args.hidden_dim_energy_net,
        energy_upper_bound=args.energy_upper_bound,
        activation=parse_activation(args.activation_energy_net),
        use_batch_norm=args.use_batch_norm_energy_net,
        dropout_probability=args.dropout_probability_energy_net
    )

    model.load_state_dict(torch.load(os.path.join(save_dir, "model"), map_location=device))

    # Create MADE
    made = ResidualMADEJoint(
        input_dim=2*dim,
        n_residual_blocks=args.n_residual_blocks_made,
        hidden_dim=args.hidden_dim_made,
        output_dim_multiplier=output_dim_multiplier,
        conditional=False,
        activation=parse_activation(args.activation_made),
        use_batch_norm=args.use_batch_norm_made,
        dropout_probability=args.dropout_probability_made
    )

    # Create proposal
    proposal = AemJointProposal(
        autoregressive_net=made,
        num_context_units=args.context_dim,
        num_components=args.n_mixture_components,
        mixture_component_min_scale=args.mixture_component_min_scale,
        apply_context_activation=args.apply_context_activation
    )

    proposal.load_state_dict(torch.load(os.path.join(save_dir, "proposal"), map_location=device)) # TODO: check so that this loads params also of made

    # create aem
    crit = AemSmcCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)
    #AemIsJointCrit

    model.eval()
    made.eval()
    crit.set_training(False)

    file = open("{}/eval_{}_set_smc_single_dim.txt".format(save_dir, 'test'), "w")
    print("{} evaluation, using SMC".format(args.dataset_name), file=file)
    print("=======================\n", file=file)

    # Importance sampling eval loop
    print("Evaluating model...")
    log_prob_unnorm_est_all_ex = []
    log_prob_proposal_all_ex = []
    batch_all_ex = []

    # Loop over batches of eval data
    n_eval_ex = test_loader.dataset.n

    with torch.no_grad():
        for b, y in enumerate(test_loader):
            # print("Batch {} of {}".format(b + 1, len(data_loader)))

            y = y.to(device)
            log_prob_est_ex, log_prob_proposal_ex = crit.unnorm_log_prob(y, return_all=True)

            # The single line above has replaced the following:
            # energy_context_curr, proposal_params_curr = sess.run(
            #    (aem.energy_context, aem.proposal_params), {x_batch: x_batch_curr}
            # )
            # for energy_context_ex, proposal_params_ex, x_batch_ex in zip(
            #        energy_context_curr, proposal_params_curr, x_batch_curr
            # ):
            #    log_prob_est_ex, log_prob_proposal_ex = sess.run(
            #        (aem.log_prob_est_data, aem.proposal_log_prob_data),
            #        {
            #            aem.energy_context: energy_context_ex[None, ...],
            #            aem.proposal_params: proposal_params_ex[None, ...],
            #            x_batch: x_batch_ex[None, ...],
            #        },
            #    )

            log_prob_unnorm_est_all_ex.append(log_prob_est_ex)
            log_prob_proposal_all_ex.append(log_prob_proposal_ex)
            batch_all_ex.append(y)

        log_normalizer = crit.log_part_fn().cpu().numpy()

        # Additionally check variance of log_normalizer
        num_est = 10
        log_norm = torch.zeros((num_est, dim))
        for i in range(num_est):
            log_norm[i, :] = crit.log_part_fn(return_all=True)
            
          
    log_prob_est_all_unnorm = torch.concat(tuple(log_prob_unnorm_est_all_ex)).cpu().numpy()
    log_proposal_est_all = torch.concat(tuple(log_prob_proposal_all_ex)).cpu().numpy()
    batch_all = torch.concat(tuple(batch_all_ex)).cpu().numpy()

    plt.hist(batch_all[:, 0], label="True", density=True)
    plt.plot(batch_all[:, 0], np.exp(log_prob_est_all_unnorm[:, 0]), 'o', label="Predicted")
    #plt.plot(batch_all[:, 0], np.exp(log_prob_est_all_unnorm[:, 0] - log_norm.mean().item()), 'o', label="Predicted norm")
    plt.plot(batch_all[:, 0], np.exp(log_proposal_est_all[:, 0]), 'o', label="Proposal")
    plt.xlabel("x")
    plt.legend()
    plt.savefig(save_dir + "/dim_zero_hist_" + args.dataset_name + "_" + args.criterion)

    log_prob_est_all = np.sum(log_prob_est_all_unnorm, axis=-1) - log_norm.sum(dim=-1).mean().cpu().numpy() # log_normalizer
    log_prob_proposal_all = np.sum(log_proposal_est_all, axis=-1)

    print(
        "Std of log normalizer sep with {} samples: {}".format(
            num_est, log_norm.cpu().numpy().std(axis=0)
        ),
        file=file)
        
    print(
        "Std of log normalizer with {} samples: {}".format(
            num_est, log_norm.sum(dim=0).cpu().numpy().std()
        ),
        file=file)

    # Compute mean, standard dev and standard error of log prob estimates
    log_prob_est_mean, log_prob_est_std = np.mean(log_prob_est_all), np.std(log_prob_est_all)
    log_prob_est_sterr = log_prob_est_std / np.sqrt(n_eval_ex)
    # Compute mean, standard dev and standard error of proposal log probs
    log_prob_proposal_mean, log_prob_proposal_std = (
        np.mean(log_prob_proposal_all),
        np.std(log_prob_proposal_all),
    )
    log_prob_proposal_sterr = log_prob_proposal_std / np.sqrt(n_eval_ex)

    # Save outputs
    print(
        "Importance sampling estimate with {} samples:".format(
            args.n_importance_samples
        ),
        file=file,
    )
    
    
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean: {}".format(log_prob_est_mean), file=file)
    print("Stddev: {}".format(log_prob_est_std), file=file)
    print("Stderr: {}\n".format(log_prob_est_sterr), file=file)

    print("Proposal log probabilities:", file=file)
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean: {}".format(log_prob_proposal_mean), file=file)
    print("Stddev: {}".format(log_prob_proposal_std), file=file)
    print("Stderr: {}\n".format(log_prob_proposal_sterr), file=file)

    file.close()

    return log_prob_est_mean, log_prob_est_std


if __name__ == '__main__':
    args = parse_args()
    main(args)
