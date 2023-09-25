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
from src.models.aem.energy_net_separate import ResidualEnergyNetSep
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z_no_context import AemJointProposalWOContext
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
        'is': ([AemIsJointCrit], ["aem_is_j_st_mm"]),
        'cis': ([AemCisJointCrit], ["aem_cis_j_st_mm"]),
        'pers': ([AemCisJointPersCrit], ["aem_cis_pers_j_st_mm"]),
        'adaptive': ([AemCisJointAdaCrit], ["aem_cis_adapt_j_st_mm"]),
        'pers_adaptive': ([AemCisJointAdaPersCrit], ["aem_cis_pers_adapt_j_st_mm"]),
        'smc': ([AemSmcCrit], ["aem_smc_j_st_mm"]),
        'csmc': ([AemSmcCondCrit], ["aem_csmc_j_st_mm"]),
        'csmc_pers': ([AemSmcCondPersCrit], ["aem_csmc_pers_j_st_mm"]),
        'smc_adaptive': ([AemSmcAdaCrit], ["aem_smc_adapt_j_st_mm"]),
        'csmc_adaptive': ([AemSmcCondAdaCrit], ["aem_csmc_adapt_j_st_mm"]),
    }

    if args.criterion in crit_dir:
        crits, crit_lab = crit_dir[args.criterion]
        
        if args.dims is not None:
            crit_lab = [crit_lab[0] + "_d_" + str(args.dims)]
        
        if args.energy_upper_bound > 0.0:
            crit_lab = [crit_lab[0] + "_ub_" + str(args.energy_upper_bound)]
            
        if args.n_mixture_components < 10:
            crit_lab = [crit_lab[0] + "_num_comp_" + str(args.n_mixture_components)]
    else:
        crits, crit_lab = [], []
        print("Unknown criterion!")
        
    # For saving q
    _, is_lab = crit_dir['is']
    
    if args.dims is not None:
        is_lab = [is_lab[0] + "_d_" + str(args.dims)]
        
    if args.n_mixture_components < 10:
        is_lab = [is_lab[0] + "_num_comp_" + str(args.n_mixture_components)]

    for i in range(args.reps):
        test_loader = load_data(data_name, args)
        #print(test_loader.dataset.n)
        #print(validation_loader.dataset.n)
        #print(train_loader.loader.dataset.n)
        #print(train_loader.loader.dataset.dim)
        save_dir_q = os.path.join(base_dir, is_lab[0] + "_q_rep_" + str(i))

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            run_test(test_loader, crit, save_dir, save_dir_q, args)
          


def load_data(name, args):
    # create datasets

    gen = None

    dataset = data_uci.load_uci_dataset(args.dataset_name, split='test', num_dims=args.dims, scaling="normalize") #, frac=args.val_frac)
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        generator=gen
    )

    return test_loader


def run_test(test_loader, criterion, save_dir, save_dir_q, args):
    # Create energy net

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')

    dim = test_loader.dataset.dim  # D
    output_dim_multiplier = args.context_dim + 3 * args.n_mixture_components  # K + 3M
    
    # Create energy net
    made_model = ResidualMADEJoint(
        input_dim=2*dim,
        n_residual_blocks=args.n_residual_blocks_made,
        hidden_dim=args.hidden_dim_made,
        output_dim_multiplier=args.context_dim,
        conditional=False,
        activation=parse_activation(args.activation_made),
        use_batch_norm=args.use_batch_norm_made,
        dropout_probability=args.dropout_probability_made
    )

    model = ResidualEnergyNetSep(
        input_dim=(args.context_dim + 1),
        made=made_model,
        n_residual_blocks=args.n_residual_blocks_energy_net,
        hidden_dim=args.hidden_dim_energy_net,
        energy_upper_bound=args.energy_upper_bound,
        activation=parse_activation(args.activation_energy_net),
        use_batch_norm=args.use_batch_norm_energy_net,
        dropout_probability=args.dropout_probability_energy_net,
        apply_context_activation=args.apply_context_activation
    )

    model.load_state_dict(torch.load(os.path.join(save_dir, "model"), map_location=device))

    # Create MADE
    made = ResidualMADEJoint(
        input_dim=2*dim,
        n_residual_blocks=args.n_residual_blocks_made,
        hidden_dim=args.hidden_dim_made,
        output_dim_multiplier=3 * args.n_mixture_components,
        conditional=False,
        activation=parse_activation(args.activation_made),
        use_batch_norm=args.use_batch_norm_made,
        dropout_probability=args.dropout_probability_made
    )

    # Create proposal
    proposal = AemJointProposalWOContext(
        autoregressive_net=made,
        num_components=args.n_mixture_components,
        mixture_component_min_scale=args.mixture_component_min_scale,
        apply_context_activation=args.apply_context_activation
    )

    # save_dir_q
    proposal.load_state_dict(torch.load(os.path.join(save_dir_q, "proposal_final"), map_location=device)) # TODO: check so that this loads params also of made

     # create aem
    crit = AemIsJointCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)

    model.eval()
    made.eval()
    crit.set_training(False)

    print("Evaluating model...")

    # Loop over batches of eval data
    n_eval_ex = test_loader.dataset.n
    
    batch_all = test_loader.dataset.data
    
    num_samp = 5
    select_ind = np.random.choice(batch_all.shape[0], size=num_samp, replace=False)
    batch_selected = torch.tensor(batch_all[select_ind, :]).to(device)
    
    min_range, max_range = -5, 5
    num_new = 1000
    y_range = torch.linspace(min_range, max_range, num_new).reshape(-1, 1)
    y_new = y_range.repeat(num_samp, 1).to(device)
    
    y = torch.repeat_interleave(batch_selected, num_new, dim=0)
    
    col_p, col_q = "#FB575D", "#8A5AC2"    
    
    with torch.no_grad():
        for i in range(dim):
            # Eval q
            net_input = torch.cat(
            (y * crit.mask[i, :], crit.mask[i, :].reshape(1, -1).repeat(y.shape[0], 1)),
            dim=-1)

            q_i, context = crit._noise_distr.forward(net_input)
            log_q_y_new = crit._noise_distr.inner_log_prob(q_i, y_new).squeeze().cpu()
            
            assert torch.allclose(context, net_input)
            
            # Eval p
            log_p_tilde_y_new = crit._model_log_probs(y_new, context).cpu()
            
            fig, ax = plt.subplots(int(np.ceil(num_samp / 5)), 5, figsize=(36, 12))
            
            for j, axis in enumerate(ax.reshape(-1)):
                if j < num_samp:
                    axis.plot(y_range[:, 0].cpu(), np.exp(log_p_tilde_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_p)
                    axis.plot(y_range[:, 0].cpu(), np.exp(log_q_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_q)
                else:
                    break
                
                axis.set_xlabel("x")
                if j == 0:
                    axis.legend(['p', 'q'])
                    
           
            plt.savefig(save_dir + "/dim_" + str(i) + "_p_q_" + args.dataset_name + "_" + args.criterion)
            plt.close()
        

if __name__ == '__main__':
    args = parse_args()
    main(args)
