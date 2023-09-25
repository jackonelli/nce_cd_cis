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

    crits = ['CSMC']
    crit_lab = ["aem_csmc_j"]
        
 
    if args.energy_upper_bound > 0.0:
        crit_lab = [cl + "_ub_" + str(args.energy_upper_bound) for cl in crit_lab]


    for i in range(args.reps):
        test_loader = load_data(data_name, args)

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            
            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            run_test(test_loader, save_dir, args)

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


def run_test(test_loader, save_dir, args):
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
    
    # Some setup
    plot_samp = 10
    batch_size = 1
    min_range, max_range = -10, 10
    num_new = 1000
    y_range = torch.linspace(min_range, max_range, num_new).reshape(-1, 1)
    y_new = y_range.repeat(plot_samp, 1).to(device)
        
    col_p, col_q = "#FB575D", "#8A5AC2"    
   
    with torch.no_grad():
        # SMC
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

        log_normalizer = torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))
        
        # Visualisation
        ind_min = torch.argmin(log_w_tilde_y_s)
        ind_max = torch.argmax(log_w_tilde_y_s)
        select_ind = torch.cat((torch.arange(0, plot_samp - 2), ind_min.unsqueeze(dim=0), ind_max.unsqueeze(dim=0))).long()
        y = torch.repeat_interleave(torch.gather(y_s[0, :, :], dim=0, index=select_ind[:, None].repeat(1, dim)), num_new, dim=0)
        
        # Eval q
        net_input = torch.cat(
        (y * crit.mask[0, :], crit.mask[0, :].reshape(1, -1).repeat(y.shape[0], 1)),
        dim=-1)

        q_i, context = crit._noise_distr.forward(net_input)
        log_q_y_new = crit._noise_distr.inner_log_prob(q_i, y_new).squeeze().cpu()
                    
        # Eval p
        log_p_tilde_y_new = crit._model_log_probs(y_new, context).cpu()
        

        fig, ax = plt.subplots(int(np.ceil(plot_samp / 5)), 5, figsize=(36, 24))
        
        for j, axis in enumerate(ax.reshape(-1)):
            if j < plot_samp:
                axis.plot(y_range[:, 0].cpu(), np.exp(log_p_tilde_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_p)
                axis.plot(y_range[:, 0].cpu(), np.exp(log_q_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_q)
                axis.plot(y_s[0, select_ind[j], 0].cpu(), np.exp(log_p_tilde_y_s[select_ind[j]].cpu()), 'o', c='green')
                axis.plot(y_s[0, select_ind[j], 0].cpu(), np.exp(log_q_y_s[select_ind[j]].cpu()), 'o', c='yellow')
                axis.set_title("w unnorm = {}".format(log_w_tilde_y_s[0, select_ind[j]]))
            else:
                break
            
            axis.set_xlabel("x")
            if j == 0:
                axis.legend(['p', 'q'])
                
       
        plt.savefig(save_dir + "/smc_test_dim_" + str(0) + "_p_q_" + args.dataset_name + "_" + args.criterion)
        plt.close()
        
        del log_p_tilde_y_s, log_q_y_s
        
        # Dim 2 to D
        for i in range(1, dim):
            # Resample
            log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
      
            ess = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach()
            ess_all[i - 1] = ess.mean(dim=0)

            resampling_inds = ess < (num_samples / 2)
            log_weight_factor = torch.zeros(log_w_tilde_y_s.shape)

            if resampling_inds.sum() > 0:
                print("Resampling")
                
                with torch.no_grad():
                    ancestor_inds = Categorical(probs=torch.exp(log_w_y_s[resampling_inds, :])).sample(
                        sample_shape=torch.Size((num_samples,))).transpose(0, 1)
                    assert ancestor_inds.shape == (batch_size, num_samples)
                 
                    y_s[resampling_inds, :, :i] = torch.gather(
                        y_s[resampling_inds, :, :i], dim=1,
                        index=ancestor_inds[:, :, None].repeat(1, 1, i))     

            sampled_inds[~resampling_inds, :, i] = torch.arange(1, num_samples + 1, dtype=torch.float).reshape(1, -1).repeat(batch_size, 1)[~resampling_inds, :]

           
           
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
                                                    
            
            del context
            
            log_w_tilde_y_s = log_weight_factor + (log_p_tilde_y_s - log_q_y_s.detach()).reshape(-1, num_samples)
                                                    
            # Visualisation
            ind_min = torch.argmin(log_w_tilde_y_s)
            ind_max = torch.argmax(log_w_tilde_y_s)
            select_ind = torch.cat((torch.arange(0, plot_samp - 2), ind_min.unsqueeze(dim=0), ind_max.unsqueeze(dim=0))).long()
            y = torch.repeat_interleave(torch.gather(y_s[0, :, :], dim=0, index=select_ind[:, None].repeat(1, dim)), num_new, dim=0)
            
            # Eval q
            net_input = torch.cat(
            (y * crit.mask[i, :], crit.mask[i, :].reshape(1, -1).repeat(y.shape[0], 1)),
            dim=-1)

            q_i, context = crit._noise_distr.forward(net_input)
            log_q_y_new = crit._noise_distr.inner_log_prob(q_i, y_new).squeeze().cpu()
                        
            # Eval p
            log_p_tilde_y_new = crit._model_log_probs(y_new, context).cpu()
            

            
            
            fig, ax = plt.subplots(int(np.ceil(plot_samp / 5)), 5, figsize=(36, 24))
            
            for j, axis in enumerate(ax.reshape(-1)):
                if j < plot_samp:
                    axis.plot(y_range[:, 0].cpu(), np.exp(log_p_tilde_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_p)
                    axis.plot(y_range[:, 0].cpu(), np.exp(log_q_y_new[(j*num_new):((j+1)*num_new)]), 'o', c=col_q)
                    axis.plot(y_s[0, select_ind[j], i].cpu(), np.exp(log_p_tilde_y_s[select_ind[j]].cpu()), 'o', c='green')
                    axis.plot(y_s[0, select_ind[j], i].cpu(), np.exp(log_q_y_s[select_ind[j]].cpu()), 'o', c='yellow')
                    axis.set_title("w unnorm = {}, {}".format(log_w_tilde_y_s[0, select_ind[j]], log_weight_factor[0, [select_ind[j]]] + log_p_tilde_y_s[select_ind[j]] - log_q_y_s[select_ind[j]]))
                else:
                    break
                
                axis.set_xlabel("x")
                if j == 0:
                    axis.legend(['p', 'q'])
                    
           
            plt.savefig(save_dir + "/smc_test_dim_" + str(i) + "_p_q_" + args.dataset_name + "_" + args.criterion)
            plt.close()
                            

            del log_p_tilde_y_s, log_q_y_s, log_weight_factor
            

            log_normalizer += torch.logsumexp(log_w_tilde_y_s, dim=1) - torch.log(torch.Tensor([num_samples]))

        log_w_y_s = log_w_tilde_y_s - torch.logsumexp(log_w_tilde_y_s, dim=1, keepdim=True)
        ess_all[-1] = torch.exp(- torch.logsumexp(2 * log_w_y_s, dim=1)).detach().mean(dim=0)



if __name__ == '__main__':
    args = parse_args()
    main(args)
