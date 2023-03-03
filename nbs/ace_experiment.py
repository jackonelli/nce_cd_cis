import torch

from src.experiments.ace_exp_utils import parse_args

from src.data.uci import UCIDataset

from src.nce.ace_is import AceIsCrit
from src.models.ace.ace_model import AceModel
from src.noise_distr.ace_proposal import AceProposal

from src.training.model_training import train_ace_model


def main(args):

    # Load data
    data_root_dir = 'src/data/datasets/uci/uci_datasets'
    data_name = "gas"

    train_loader, validation_loader, test_loader = load_data(data_name, data_root_dir)

    save_dir = "nbs/res/ace/"
    run_experiment(train_loader, validation_loader, save_dir, args)

    ll = run_test(test_loader, save_dir, args)

    print("Test log. likelihood: {}".format(ll))


def load_data(name, root_dir, batch_size=32):
    train_data = UCIDataset(name=name, set="train", root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = UCIDataset(name=name, set="val", root_dir=root_dir)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)

    test_data = UCIDataset(name=name, set="test", root_dir=root_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def run_experiment(train_loader, validation_loader, save_dir, args):

    # Model specs
    num_features = train_loader.dataset.num_features
    model = AceModel(num_features=num_features, num_context_units=args.num_context_units)
    proposal = AceProposal(num_features=num_features, num_context_units=args.num_context_units)
    crit = AceIsCrit(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg)

    train_ace_model(crit, train_loader, validation_loader, save_dir, weight_decay=0.0, decaying_lr=True,
                    num_epochs=args.num_epochs, lr=args.lr, scheduler_opts=(args.num_epochs_decay, args.lr_factor),
                    device=torch.device(args.device))


def run_test(test_loader, save_dir, args):

    device = torch.device(args.device)

    num_features = test_loader.dataset.num_features
    model = AceModel(num_features=num_features, num_context_units=args.num_context_units)
    model.load_state_dict(torch.load(save_dir + "_model"))

    proposal = AceProposal(num_features=num_features, num_context_units=args.num_context_units)
    proposal.load_state_dict(torch.load(save_dir + "_proposal"))

    crit = AceIsCrit(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg)

    model, proposal = model.to(device), proposal.to(device)
    ll = 0
    for (y, idx_) in test_loader:
        y = y.to(device)
        ll += crit.log_likelihood(y, args.num_is_samples)

    return ll


if __name__ == '__main__':
    args = parse_args()
    main(args)

