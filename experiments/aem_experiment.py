import json
import os
import numpy as np
import torch
from torch.utils import data

from src.aem.aem_is_joint_z import AemIsJointCrit
from src.aem.aem_cis_joint_z import AemCisJointCrit
from src.aem.aem_smc_cond import AemSmcCondCrit

from src.models.aem.energy_net import ResidualEnergyNet
from src.models.aem.made_joint_z import ResidualMADEJoint
from src.noise_distr.aem_proposal_joint_z import AemJointProposal

from src.data import data_uci
from src.data.data_uci.uciutils import get_project_root

from src.experiments.aem_exp_utils import parse_activation, parse_args, InfiniteLoader
from src.training.model_training import train_aem_model


def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_name = args.dataset_name
    proj_dir = os.path.join(get_project_root(), "deep_ext_obj/experiments/res/aem/")
    base_dir = os.path.join(proj_dir, args.dataset_name)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)  # (io.get_checkpoint_root())

    crit_dict = {
        'is': ([AemIsJointCrit], ["aem_is_j"]),
        'cis': ([AemCisJointCrit], ["aem_cis_j"]),
        'csmc': ([AemSmcCondCrit], ["aem_csmc_j"])
    }

    if args.criterion in crit_dict:
        crits, crit_lab = crit_dict[args.criterion]
        
        if args.dims is not None:
            crit_lab = [crit_lab[0] + "_d_" + str(args.dims)]
            
        if args.energy_upper_bound > 0.0:
            crit_lab = [crit_lab[0] + "_ub_" + str(args.energy_upper_bound)]
            
        if args.n_mixture_components < 10:
            crit_lab = [crit_lab[0] + "_num_comp_" + str(args.n_mixture_components)]

        
    else:
        crits, crit_lab = [], []
        print("Unknown criterion!")

    for i in range(args.reps):
        train_loader, validation_loader, test_loader = load_data(data_name, args)

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            save_dir = os.path.join(base_dir, lab + "_rep_" + str(i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)  # (io.get_checkpoint_root())

            run_train(train_loader, validation_loader, crit, save_dir, args)
            print("Training complete")

def load_data(name, args):
    # create datasets

    gen = None
    if args.use_gpu and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        gen = torch.Generator(device='cuda')
    else:
        gen = torch.Generator(device='cpu')

    # training set
    train_dataset = data_uci.load_uci_dataset(name, split='train', num_dims=args.dims)
    train_loader = InfiniteLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_epochs=None,
        use_gpu=args.use_gpu
    )

    # validation set
    val_dataset = data_uci.load_uci_dataset(name, split='val', frac=args.val_frac, num_dims=args.dims)
    validation_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=gen
    )

    test_dataset = data_uci.load_uci_dataset(args.dataset_name, split='test', num_dims=args.dims)
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        generator=gen
    )

    return train_loader, validation_loader, test_loader
    
    
def load_models(dim, args):

    # define parameters for MADE and energy net
    output_dim_multiplier = args.context_dim + 3 * args.n_mixture_components  # K + 3M

    # Create energy net
    model = ResidualEnergyNet(
        input_dim=(args.context_dim + 1),
        n_residual_blocks=args.n_residual_blocks_energy_net,
        hidden_dim=args.hidden_dim_energy_net,
        energy_upper_bound=args.energy_upper_bound,
        activation=parse_activation(args.activation_energy_net),
        use_batch_norm=args.use_batch_norm_energy_net,
        dropout_probability=args.dropout_probability_energy_net
    )

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
    
    return model, made, proposal


def run_train(train_loader, validation_loader, criterion, save_dir, args):

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dim = train_loader.loader.dataset.dim  # D
    model, made, proposal = load_models(dim, args)
    

    # create aem
    crit = criterion(model, proposal, args.n_proposal_samples_per_input, args.n_proposal_samples_per_input_validation)
    #crit.set_training(True)

    filename = save_dir + '/config.json'
    with open(filename, 'w') as file:
        json.dump(vars(args), file)

    train_aem_model(crit, train_loader, validation_loader, save_dir, decaying_lr=True,
                    num_training_steps=args.n_total_steps, num_warm_up_steps=args.alpha_warm_up_steps, hard_warmup=args.hard_alpha_warm_up,
                    lr=args.learning_rate, validation_freq=args.monitor_interval, device=device)



if __name__ == '__main__':
    args = parse_args()
    main(args)
