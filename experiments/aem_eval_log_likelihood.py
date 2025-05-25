import json
import os
import numpy as np
import torch
from torch.utils import data

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.aem.aem_smc import AemSmcCrit
from src.aem.aem_smc_cond import AemSmcCondCrit

from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal

from src.data import data_uci
from src.data.data_uci.uciutils import get_project_root

from src.utils.aem_exp_utils import parse_activation, parse_args, InfiniteLoader, standard_resampling
from src.training.model_training import train_aem_model
from experiments.aem import load_models


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_name = args.dataset_name
    proj_dir = os.path.join(get_project_root(), "nce_cd_cis/experiments/res/aem/")
    base_dir = os.path.join(proj_dir, args.dataset_name)

    if args.dataset_name in ["miniboone", "bsds300"]:
        crits = ['CIS', 'CSMC']
        crit_lab = ["aem_cis_j", "aem_csmc_j"]
    else:
        crits = ['IS', 'CIS', 'CSMC']
        crit_lab = ["aem_is_j", "aem_cis_j", "aem_csmc_j"]
        
    file_dir = os.path.join(base_dir, 'all')

    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  # (io.get_checkpoint_root())
    
    file = open("{}/eval_{}_set_{}.txt".format(file_dir, args.val_split, str(args.n_importance_samples)), "w")
    for i in range(args.reps):
        test_loader = load_data(data_name, args)
        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            print("-------------------------------------------------\n", file=file)
            print("Criterion: {}".format(crit), file=file)

            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            print("Evaluating model from {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            ll_mean, ll_sterr = run_test(test_loader, file, save_dir, args)
            print("Test log. likelihood, mean: {}".format(ll_mean))
            print("Test log. likelihood, sterr: {}".format(ll_sterr))

    file.close()

def load_data(name, args):
    # create dataset
    gen = None
    if args.use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gen = torch.Generator(device='cuda')
    else:
        gen = torch.Generator(device='cpu')
    
    dataset = data_uci.load_uci_dataset(args.dataset_name, split=args.val_split, frac=args.val_frac) 
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        generator=gen
    )

    return test_loader


def run_test(test_loader, file, save_dir, args):
    # Create energy net

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    dim = test_loader.dataset.dim  # D
    model, made, proposal = load_models(dim, args)

    model.load_state_dict(torch.load(os.path.join(save_dir, "model"), map_location=device))
    proposal.load_state_dict(torch.load(os.path.join(save_dir, "proposal"), map_location=device)) 

    # create aem
    crit = AemIsJointCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)
    crit_smc = AemSmcCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)

    model.eval()
    made.eval()
    crit.set_training(False)
    crit_smc.set_training(False)
    crit_smc.set_resampling_function(standard_resampling) 
   
    # Importance sampling eval loop
    print("Evaluating model...")
    log_prob_unnorm_est_all_ex = []
    log_prob_proposal_all_ex = []

    # Loop over batches of eval data
    n_eval_ex = test_loader.dataset.n
    num_est = 10
    with torch.no_grad():
        for b, y in enumerate(test_loader):
            # print("Batch {} of {}".format(b + 1, len(data_loader)))

            y = y.to(device)
            log_prob_est_ex, log_prob_proposal_ex = crit.unnorm_log_prob(y)


            log_prob_unnorm_est_all_ex.append(log_prob_est_ex)
            log_prob_proposal_all_ex.append(log_prob_proposal_ex)

        # Estimate log_normalizer with is
        log_norm = torch.zeros((num_est,))
        if args.n_importance_samples < 1e5: # Arbitrary threshold to avoid memory issues
            for i in range(num_est):
                log_norm[i], _ = crit.log_part_fn(return_ess=True)
            
        # Estimate log_normalizer with smc
        log_norm_smc = torch.zeros((num_est,))
        for i in range(num_est):
            log_norm_smc[i], _ = crit_smc.log_part_fn(return_ess=True)
            

    log_prob_est_all = torch.concat(tuple(log_prob_unnorm_est_all_ex)).cpu().numpy() 
    log_prob_proposal_all = torch.concat(tuple(log_prob_proposal_all_ex)).cpu().numpy()
    log_norm, log_norm_smc = log_norm.cpu().numpy(), log_norm_smc.cpu().numpy()

    # Compute mean, standard dev and standard error of log prob estimates using is
    log_prob_est_mean, log_prob_est_std = np.mean(log_prob_est_all) - np.mean(log_norm), np.std(log_norm)
    log_prob_est_sterr = log_prob_est_std / np.sqrt(num_est)
        
    # Compute mean, standard dev and standard error of log prob estimates using smc
    log_prob_est_mean_smc, log_prob_est_std_smc = np.mean(log_prob_est_all) - np.mean(log_norm_smc), np.std(log_norm_smc)
    log_prob_est_sterr_smc = log_prob_est_std_smc / np.sqrt(num_est)
    
    # Compute mean of proposal log probs
    log_prob_proposal_mean = np.mean(log_prob_proposal_all)

    # Save outputs
    print(
        "Importance sampling estimate with {} particles:".format(
            args.n_importance_samples
        ),
        file=file,
    )
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean, IS: {}".format(log_prob_est_mean), file=file)
    print("Stddev, IS: {}".format(log_prob_est_std), file=file)
    print("Stderr, IS: {}\n".format(log_prob_est_sterr), file=file)
    print("Mean, SMC: {}".format(log_prob_est_mean_smc), file=file)
    print("Stddev, SMC: {}".format(log_prob_est_std_smc), file=file)
    print("Stderr, SMC: {}\n".format(log_prob_est_sterr_smc), file=file)

    print("Proposal log probabilities:", file=file)
    print("-------------------------------------------------\n", file=file)
    print("No. examples: {}".format(n_eval_ex), file=file)
    print("Mean: {}".format(log_prob_proposal_mean), file=file)

    return log_prob_est_mean_smc, log_prob_est_sterr_smc


if __name__ == '__main__':
    args = parse_args()
    main(args)
