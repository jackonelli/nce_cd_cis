import torch

from src.experiments.ace_exp_utils import parse_args

from src.data.uci import UCIDataset

from src.nce.ace_is import AceIsCrit
from src.models.ace.ace_model import AceModel
from src.noise_distr.ace_proposal import AceProposal

from src.training.model_training import train_ace_model
from src.experiments.ace_exp_utils import UniformMaskGenerator

def main(args):

    # Load data
    data_root_dir = 'src/data/datasets/uci/'
    data_name = "gas"

    train_loader, validation_loader, test_loader = load_data(data_name, data_root_dir, args)

    save_dir = "nbs/res/ace/"
    run_train(train_loader, validation_loader, save_dir, args)

    ll = run_test(test_loader, save_dir, args)

    print("Test log. likelihood: {}".format(ll))


def load_data(name, root_dir, args):
    train_data = UCIDataset(name=name, set="train", root_dir=root_dir, noise_scale=args.noise_scale)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    validation_data = UCIDataset(name=name, set="val", root_dir=root_dir)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size)

    test_data = UCIDataset(name=name, set="test", root_dir=root_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    return train_loader, validation_loader, test_loader


def run_train(train_loader, validation_loader, save_dir, args):

    # Model specs
    num_features = train_loader.dataset.num_features
    model = AceModel(num_features=num_features, num_context_units=args.num_context_units,
                     dropout_rate=args.dropout_rate)

    proposal = AceProposal(num_features=num_features, num_context_units=args.num_context_units,
                           num_blocks=args.proposal_num_blocks, num_hidden_units=args.proposal_num_hidden,
                           dropout_rate=args.dropout_rate)
    crit = AceIsCrit(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg,
                     device=torch.device(args.device))

    train_ace_model(crit, train_loader, validation_loader, save_dir, weight_decay=0.0, decaying_lr=True,
                    num_training_steps=args.num_training_steps, num_warm_up_steps=args.num_warm_up_steps,
                    lr=args.lr, scheduler_opts=(args.num_steps_decay, args.lr_factor),
                    device=torch.device(args.device))


def run_test(test_loader, save_dir, args):

    device = torch.device(args.device)

    num_features = test_loader.dataset.num_features
    model = AceModel(num_features=num_features, num_context_units=args.num_context_units,
                     dropout_rate=args.dropout_rate)
    model.load_state_dict(torch.load(save_dir + "_model"))

    proposal = AceProposal(num_features=num_features, num_context_units=args.num_context_units,
                           num_blocks=args.proposal_num_blocks, num_hidden_units=args.proposal_num_hidden,
                           dropout_rate=args.dropout_rate)
    proposal.load_state_dict(torch.load(save_dir + "_proposal"))
    model, proposal = model.to(device), proposal.to(device)

    crit = AceIsCrit(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg,
                     device=torch.device(args.device))

    ll = 0
    for (y, idx_) in test_loader:
        y = y.to(device)
        ll += crit.log_likelihood(y, args.num_is_samples)

    return ll




if __name__ == '__main__':
    args = parse_args()
    main(args)

