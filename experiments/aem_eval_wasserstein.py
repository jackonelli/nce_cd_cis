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

from src.utils.aem_exp_utils import parse_activation, parse_args, InfiniteLoader, wasserstein_metric, standard_resampling
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

    
    if args.dims is not None:
        crit_lab = [cl + "_d_" + str(args.dims) for cl in crit_lab]
        file_dir = file_dir + "_ub_" +  "_d_" + str(args.dims)

 
    if args.energy_upper_bound > 0.0:
        crit_lab = [cl + "_ub_" + str(args.energy_upper_bound) for cl in crit_lab]
        file_dir = file_dir + "_ub_" + str(args.energy_upper_bound)
        
    if args.n_mixture_components < 10:
        crit_lab = [crit_lab[0] + "_num_comp_" + str(args.n_mixture_components)]
        file_dir = file_dir + "_num_comp_" + str(args.n_mixture_components)
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  # (io.get_checkpoint_root())
    
    file = open("{}/eval_wasser_{}_set_{}.txt".format(file_dir, args.val_split, str(args.n_importance_samples)), "w")
    for i in range(args.reps):
        test_loader = load_data(data_name, args)
        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            print("-------------------------------------------------\n", file=file)
            print("Criterion: {}".format(crit), file=file)

            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            print("Evaluating model from {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            dist_mean, dist_sterr = run_test(test_loader, file, save_dir, args)
            print("Wasserstein dist, mean: {}".format(dist_mean))
            print("Wasserstein dist, sterr: {}".format(dist_sterr))

    file.close()

def load_data(name, args):
    # create dataset
    gen = None
    if args.use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gen = torch.Generator(device='cuda')
    else:
        gen = torch.Generator(device='cpu')
    
    dataset = data_uci.load_uci_dataset(args.dataset_name, split=args.val_split)#, frac=args.val_frac)
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
    output_dim_multiplier = args.context_dim + 3 * args.n_mixture_components  # K + 3M

    dim = test_loader.dataset.dim  # D
    model, made, proposal = load_models(dim, args)
    
    model.load_state_dict(torch.load(os.path.join(save_dir, "model"), map_location=device))
    proposal.load_state_dict(torch.load(os.path.join(save_dir, "proposal"), map_location=device)) 

    # create aem
    crit = AemSmcCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)

    model.eval()
    made.eval()
    crit.set_training(False)
    crit.set_resampling_function(standard_resampling)
   
    # Importance sampling eval loop
    print("Evaluating model...")
    
   
    num_samp = args.n_importance_samples  # Reusing this one
    num_est = 10
    dist = torch.zeros((num_est,))
            
    for i in range(num_est):
        # Sample from data
        data_samples_all = test_loader.dataset.data

        select_ind = np.random.choice(data_samples_all.shape[0], size=num_samp, replace=True) # replace = False
        x_samples = torch.tensor(data_samples_all[select_ind, :]).cpu()
        
        # Sample from model (using SMC)
        with torch.no_grad():
            _, y_s, log_w_tilde_y_s, _ = crit.inner_smc(1, num_samp, None)

        select_ind = torch.distributions.Categorical(logits=log_w_tilde_y_s.squeeze(dim=0)).sample((num_samp,))
        y_samples = torch.gather(y_s.squeeze(dim=0), dim=0, index=select_ind[:, None].repeat(1, dim)).cpu()

        dist[i] = wasserstein_metric(x_samples, y_samples)

          
    # Save outputs
    print(
        "Wasserstein distance with {} samples:".format(
            num_samp
        ),
        file=file,
    )
    
    dist_mean, dist_std = dist.mean(), dist.std()
    
    print(
        "Mean: {}".format(
            dist_mean
        ),
        file=file,
    )
    
    print(
        "Stddev: {}".format(
            dist_std
        ),
        file=file,
    )
    
    print(
        "Stderr: {}".format(
            dist_std / np.sqrt(num_est)
        ),
        file=file,
    )
    
    return dist_mean, dist_std / np.sqrt(num_est)


if __name__ == '__main__':
    args = parse_args()
    main(args)
