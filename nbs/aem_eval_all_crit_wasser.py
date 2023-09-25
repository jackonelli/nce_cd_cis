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

from nbs.wasserstein import wasserstein_metric

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_name = args.dataset_name
    proj_dir = os.path.join(get_project_root(), "deep_ext_obj/nbs/res/aem_backup/") #os.path.join(get_project_root(), "deep_ext_obj/nbs/res/aem/")
    base_dir = os.path.join(proj_dir, args.dataset_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # (io.get_checkpoint_root())

    # Save args
    #with open(os.path.join(base_dir, 'commandline_args.txt'), 'w') as f:
    #   json.dump(args.__dict__, f, indent=2)

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
    
    file = open("{}/eval_wasser_{}_set_large.txt".format(file_dir, 'test'), "w")
    for i in range(args.reps):
        test_loader = load_data(data_name, args)
        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            print("-------------------------------------------------\n", file=file)
            print("Criterion: {}".format(crit), file=file)

            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            print("Evaluating model from {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            dist = run_test(test_loader, file, save_dir, args)
            print("Wasserstein dist: {}".format(dist))

    file.close()

def load_data(name, args):
    # create dataset
    gen = None
    if args.use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gen = torch.Generator(device='cuda')
    else:
        gen = torch.Generator(device='cpu')
    
    dataset = data_uci.load_uci_dataset(args.dataset_name, split='test') #, frac=args.val_frac)
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

    model.eval()
    made.eval()
    crit.set_training(False)
   
    # Importance sampling eval loop
    print("Evaluating model...")
    
    
   
    num_samp = args.n_importance_samples  # Sorry, hackar denna lite
    reps = 10
    dist = torch.zeros((reps,))
            
    for r in range(reps):
        # Sample from data
        data_samples_all = test_loader.dataset.data


        select_ind = np.random.choice(data_samples_all.shape[0], size=num_samp, replace=False)
        x_samples = torch.tensor(data_samples_all[select_ind, :]).cpu()
        
        # Sample from model (using SMC)
        with torch.no_grad():
            _, y_s, log_w_tilde_y_s, _ = crit.inner_smc(1, num_samp, None)

        select_ind = torch.distributions.Categorical(logits=log_w_tilde_y_s.squeeze(dim=0)).sample((num_samp,))
        y_samples = torch.gather(y_s.squeeze(dim=0), dim=0, index=select_ind[:, None].repeat(1, dim)).cpu()

        dist[r] = wasserstein_metric(x_samples, y_samples)

          
    # Save outputs
    print(dist)
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
        "Std: {}".format(
            dist_std
        ),
        file=file,
    )
    
    return dist_mean


if __name__ == '__main__':
    args = parse_args()
    main(args)
