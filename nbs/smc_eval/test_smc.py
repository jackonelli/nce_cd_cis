import json
import os
import numpy as np
import torch
from torch.distributions import Categorical
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

    crits = ['CIS']
    crit_lab = ["aem_cis_j_st"]
         
    if args.energy_upper_bound > 0.0:
        crit_lab = [cl + "_ub_" + str(args.energy_upper_bound) for cl in crit_lab]

    # For loading q
    is_lab = ["aem_is_j_st"]
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  # (io.get_checkpoint_root())
    
    for i in range(args.reps):
        test_loader = load_data(data_name, args)
        save_dir_q = os.path.join(base_dir, is_lab[0] + "_q_rep_" + str(i))

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
         
            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            run_test(test_loader, save_dir, save_dir_q, args)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

def load_data(name, args):
    # create dataset
    gen = None
    if args.use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gen = torch.Generator(device='cuda')
    else:
        gen = torch.Generator(device='cpu')
    
    dataset = data_uci.load_uci_dataset(args.dataset_name, split='test')#, frac=args.val_frac)
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        generator=gen
    )

    return test_loader


def run_test(test_loader, save_dir, save_dir_q, args):
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
    
    proposal.load_state_dict(torch.load(os.path.join(save_dir_q, "proposal_final"), map_location=device)) 
    
    # create aem
    crit = AemSmcCrit(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)

    model.eval()
    made.eval()
    crit.set_training(False)
   
    # SMC
    plot_dim = 5
    batch_size = 1
    num_samples = args.n_importance_samples
    y_s = torch.zeros((batch_size, num_samples, dim))
    ess_all = torch.zeros((dim,))

    sampled_inds = torch.zeros((batch_size, num_samples, dim))
    sampled_inds[:, :, 0] = torch.arange(1, num_samples + 1, dtype=torch.float).reshape(1, -1).repeat(batch_size, 1)

    # First dim
    # Propagate
    log_q_y_s, context, y_s = crit._proposal_log_probs(y_s.reshape(-1, dim), 0, num_observed=0)
    context, y_s = context.reshape(-1, num_samples, crit.num_context_units), y_s.reshape(-1, num_samples, dim)


    # Reweight
    log_p_tilde_y_s = crit._model_log_probs(y_s[:, :, 0].reshape(-1, 1),
                                            context.reshape(-1, crit.num_context_units))
    del context

    log_w_tilde_y_s = (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_samples)
    del log_p_tilde_y_s, log_q_y_s

    # Dim 2 to D
    log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))

    for i in range(1, dim):
        # Resample
        log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
  
        ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach()
        ess_all[i - 1] = ess.mean(dim=0)

        resampling_inds = ess < (num_samples / 2)
        log_weight_factor = torch.zeros(log_w_tilde_y_s.shape)

        w = log_w_y_s.clone()
        #print(log_w_tilde_y_s)
        if resampling_inds.sum() > 0:
            print("Resampling")
            
            with torch.no_grad():
                ancestor_inds = Categorical(probs=torch.exp(log_w_y_s[resampling_inds, :])).sample(
                    sample_shape=torch.Size((num_samples,))).transpose(0, 1)
                assert ancestor_inds.shape == (batch_size, num_samples)
                #print(torch.exp(log_w_y_s))
                #inds = Categorical(logits=log_w_tilde_y_s[resampling_inds, :]).sample(
                #sample_shape=torch.Size((10000,))).transpose(0, 1)
                #print(torch.bincount(inds.squeeze(), minlength=num_samples) / 10000)
                    
                y_s_ref = y_s.clone()
                y_s[resampling_inds, :, :i] = torch.gather(
                    y_s[resampling_inds, :, :i], dim=1,
                    index=ancestor_inds[:, :, None].repeat(1, 1, i))
                    
                w[resampling_inds, :] = torch.gather(log_w_y_s[resampling_inds, :], dim=1,
                        index=ancestor_inds[:, :])
                sampled_inds[resampling_inds, :, i] = ancestor_inds.float() + 1.0
                if i <= plot_dim:
                    print(i)
                    print("Test")
                    print(y_s[0, 1, :])
                    print(y_s_ref[0, ancestor_inds[0, 1], :])
                    
                y_s_old = y_s_ref.clone()
                for j in range(batch_size):
                    for k in range(num_samples):
                        if resampling_inds[j]:
                            for l in range(i):
                                y_s_ref[j, k, l] = y_s_old[j, ancestor_inds[j, k], l] 
                
                assert torch.allclose(y_s, y_s_ref)

        sampled_inds[~resampling_inds, :, i] = torch.arange(1, num_samples + 1, dtype=torch.float).reshape(1, -1).repeat(batch_size, 1)[~resampling_inds, :]

        if i <= plot_dim:
            for j in range(num_samples): # There is probably an easier way to adapt the marker size
                #print(torch.exp(log_w_y_s[:, j]))
                plt.plot(i, j + 1, '.', markersize=100*torch.exp(log_w_y_s[:, j]), c='blue', alpha=0.5)
                plt.plot(i+1, j + 1, '.', markersize=100 * torch.exp(w[:, j]), c='red', alpha=0.5)
                plt.plot(np.array([i, i + 1]), np.array([sampled_inds[:, j, i].squeeze().cpu(), j + 1]), '-.', c='orange')
       
        # if resampling_inds.sum() < batch_size:
        log_weight_factor[~resampling_inds, :] = log_w_y_s[~resampling_inds, :] + torch.log(
            torch.Tensor([num_samples]))
        del log_w_y_s, ess, resampling_inds

        # Propagate
        log_q_y_s, context, y_s = crit._proposal_log_probs(y_s.reshape(-1, dim), i, num_observed=0)
        context, y_s = context.reshape(-1, num_samples, crit.num_context_units), y_s.reshape(-1, num_samples,
                                                                                             dim)
        

        # Reweight
        log_p_tilde_y_s = crit._model_log_probs(y_s[:, :, i].reshape(-1, 1),
                                                context.reshape(-1, crit.num_context_units))
        
        print(i)        
        #print(torch.exp(log_p_tilde_y_s))
        #print(torch.exp(log_q_y_s))
                        
        log_q_y_s_ref = torch.zeros((batch_size, num_samples))             
        log_p_tilde_y_s_ref = torch.zeros((batch_size, num_samples))
        
        for j in range(batch_size):
            for k in range(num_samples):
                net_input = torch.cat(
                (y_s[j, k, :].reshape(1, -1) * crit.mask[i, :], crit.mask[i, :].reshape(1, -1)),
                dim=-1)
                log_q_y_s_ref[j, k] = crit._noise_distr.log_prob(y_s[j, k, i].reshape(1, 1), net_input)
                log_p_tilde_y_s_ref[j, k] = crit._model_log_probs(y_s[j, k, i].reshape(1, 1),
                                                net_input.reshape(1, crit.num_context_units))
        
        assert torch.abs(log_p_tilde_y_s - log_p_tilde_y_s_ref.reshape(batch_size, num_samples)).max() <= 1e-4
        assert torch.allclose(log_p_tilde_y_s, log_p_tilde_y_s_ref.reshape(batch_size, num_samples), atol=1e-4)
        assert torch.abs(log_q_y_s - log_q_y_s_ref.reshape(batch_size, num_samples)).max() <= 1e-3
        assert torch.allclose(log_q_y_s, log_q_y_s_ref.reshape(batch_size, num_samples), atol=1e-3) # TODO: varför plöstligt så liten marginal?
        del context
        
        log_w_tilde_y_s = log_weight_factor + (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_samples)
        log_w_tilde_y_s_ref = log_weight_factor + log_p_tilde_y_s_ref - log_q_y_s_ref
        assert torch.abs(log_w_tilde_y_s - log_w_tilde_y_s_ref.reshape(batch_size, num_samples)).max() <= 1e-3
        assert torch.allclose(log_w_tilde_y_s, log_w_tilde_y_s_ref.reshape(batch_size, num_samples), atol=1e-3)

        ind = torch.argmin(log_w_tilde_y_s.squeeze())
        print(log_w_tilde_y_s[0, ind])
        print(log_w_tilde_y_s_ref[0, ind])
        print(log_p_tilde_y_s[ind])
        print(log_q_y_s[ind])
        
        del log_p_tilde_y_s, log_q_y_s, log_weight_factor
        

        log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))

    log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
    ess_all[-1] = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach().mean(dim=0)

    if dim <= plot_dim:
        for j in range(num_samples):  # There is probably an easier way to adapt the marker size
            plt.plot(dim, j + 1, '.', markersize=100*torch.exp(log_w_y_s[:, j]), c='blue', alpha=0.5)
        
    print("Saving figure to {}".format(save_dir + "/smc_fig.png"))
    plt.savefig(save_dir + "/smc_fig")


if __name__ == '__main__':
    args = parse_args()
    main(args)
