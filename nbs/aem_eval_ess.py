import json
import os
import numpy as np
import torch
from torch.utils import data

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

            run_test(test_loader, crit, save_dir, args)
          

def load_data(name, args):
    # create datasets

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
    crit = criterion(model, proposal, args.n_proposal_samples_per_input, args.n_importance_samples)

    model.eval()
    made.eval()
    crit.set_training(False)

    # Importance sampling eval loop
    print("Evaluating model...")

    # Loop over batches of eval data
    n_eval_ex = test_loader.dataset.n
            
    # Test log. normalizer
    with torch.no_grad():
        if args.criterion in ['is']:
            _, ess = crit.log_part_fn(return_ess=True)
            print(ess)
        else:
            ind = np.random.randint(n_eval_ex)
            y = torch.tensor(test_loader.dataset.data[ind, :].reshape(1, -1))
            _, ess = crit.log_part_fn(y, return_ess=True)
            print(ess)
            
          
    if args.criterion in ['csmc', 'csmc_pers']:
        num_steps = 10
        update_freq = torch.zeros((dim,))
        ind = np.random.randint(n_eval_ex)
        y = torch.tensor(test_loader.dataset.data[ind, :].reshape(1, -1))
        with torch.no_grad():
            for t in range(num_steps):
                _, y_s, log_w_tilde_y_s, ess = crit.inner_smc(1, args.n_importance_samples, y)
                #print(ess)
                y_old = y.clone()
                sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s).sample()
                y = torch.gather(y_s, dim=1, index=sampled_idx[:, None, None].repeat(1, 1, y.shape[-1])).squeeze(dim=1)
                update_freq += not_close(y, y_old).squeeze() / num_steps
                
        print('Update freq.: {}'.format(update_freq))
        
    elif args.criterion in ['cis', 'pers']:
        num_steps = 100
        update_freq = torch.zeros((dim,))
        ind = np.random.randint(n_eval_ex)
        y = torch.tensor(test_loader.dataset.data[ind, :].reshape(1, -1))
        with torch.no_grad():
            for t in range(num_steps):
                _, _, _, log_w_tilde_y_s, _, y_samples = crit._log_probs(y, args.n_importance_samples)
                y_s = torch.cat((y.reshape(-1, 1, dim), y_samples.reshape(-1, args.n_importance_samples, dim)), dim=1)
                y_old = y.clone()
                sampled_idx = torch.distributions.Categorical(logits=log_w_tilde_y_s).sample()
                y = torch.gather(y_s, dim=1, index=sampled_idx[:, None, None].repeat(1, 1, y.shape[-1])).squeeze(dim=1)
                update_freq += not_close(y, y_old).squeeze() / num_steps
                print(not_close(y, y_old).squeeze() / num_steps)
        
        print(update_freq)
    #log_norm = log_norm.cpu().numpy()
                
    #plt.errorbar(np.array(num_neg), log_norm.mean(axis=-1), yerr=log_norm.std(axis=-1))
    #plt.savefig("log_normalizer_" + args.dataset_name + "_" + args.criterion)

def not_close(y, y_p, atol=1e-8, rtol=1e-5):
    return ~(torch.abs(y-y_p) < atol + rtol * torch.abs(y_p))


if __name__ == '__main__':
    args = parse_args()
    main(args)
