import torch

from src.experiments.ace_exp_utils import parse_args

from src.data.uci import UCIDataset

from src.ace.ace_is import AceIsCrit
from src.ace.ace_cis import AceCisCrit
from src.ace.ace_cis_alt import AceCisAltCrit
from src.ace.ace_cis_adapt import AceCisAdaCrit
from src.ace.ace_pers_cis import AceCisPers
from src.models.ace.ace_model import AceModel
from src.noise_distr.ace_proposal import AceProposal

from src.training.model_training import train_ace_model


def main(args):

    # Load data
    data_root_dir = 'src/data/datasets/uci/'
    data_name = args.dataset

    base_dir = "nbs/res/ace/"

    crits = [AceIsCrit]#, AceCisCrit]#, AceCisPers]
    crit_lab = ["ace_is_hm"]#, "ace_cis"]#, "ace_cis_pers"]

    ll, ll_std = torch.zeros((args.reps, len(crits))), torch.zeros((args.reps, len(crits)))
    for i in range(args.reps):
        train_loader, validation_loader, test_loader = load_data(data_name, data_root_dir, args)

        for j, (crit, lab) in enumerate(zip(crits, crit_lab)):
            save_dir = base_dir + lab + "_rep_" + str(i) + "_"

            run_train(train_loader, validation_loader, crit, save_dir, args)
            ll[i, j], ll_std[i, j] = run_test(test_loader, crit, save_dir, args)
            print("Test log. likelihood, mean: {}".format(ll[i, j]))
            print("Test log. likelihood, std: {}".format(ll_std[i, j]))


def load_data(name, root_dir, args):
    train_data = UCIDataset(name=name, set="train", root_dir=root_dir, noise_scale=args.noise_scale)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    validation_data = UCIDataset(name=name, set="val", root_dir=root_dir)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, num_workers=4)

    test_data = UCIDataset(name=name, set="test", root_dir=root_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=4)  # TODO: can't handle the amount of samples in IS  
  
    return train_loader, validation_loader, test_loader


def run_train(train_loader, validation_loader, criterion, save_dir, args):

    # Model specs
    num_features = train_loader.dataset.num_features
    model = AceModel(num_features=num_features, num_context_units=args.num_context_units,
                     dropout_rate=args.dropout_rate)

    proposal = AceProposal(num_features=num_features, num_context_units=args.num_context_units,
                           num_blocks=args.proposal_num_blocks, num_hidden_units=args.proposal_num_hidden,
                           dropout_rate=args.dropout_rate)
    crit = criterion(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg,
                     device=torch.device(args.device), batch_size=args.batch_size)

    train_ace_model(crit, train_loader, validation_loader, save_dir, weight_decay=0.0, decaying_lr=True,
                    num_training_steps=args.num_training_steps, num_warm_up_steps=args.num_warm_up_steps,
                    lr=args.lr, scheduler_opts=(args.num_steps_decay, args.lr_factor),
                    device=torch.device(args.device))


def run_test(test_loader, criterion, save_dir, args):

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

    crit = criterion(model, proposal, args.num_negative, alpha=args.alpha, energy_reg=args.energy_reg,
                     device=device)

    ll = torch.zeros((args.num_permutations,)).to(device)
    # TODO: Use same observed mask for all models? And same random permutations? (In that case I will need to generate term here and send them into the ll function)
    for (y, idx_) in test_loader:
        y = y.to(device)
        ll = ll + crit.log_likelihood(y, args.num_is_samples, args.num_permutations) * y.shape[0]

    return (ll / test_loader.dataset.num_samples).mean(), (ll / test_loader.dataset.num_samples).std()


if __name__ == '__main__':
    args = parse_args()
    main(args)

