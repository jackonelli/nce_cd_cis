import os
from typing import Optional, Tuple
import torch
import numpy as np

from src.training.training_utils import no_stopping, PolynomialLr
from src.experiments.ace_exp_utils import get_ace_losses
from src.experiments.aem_exp_utils import get_aem_losses

from tensorboardX import SummaryWriter
from tqdm import tqdm


def train_model(
    criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    weight_decay=0.0,
    decaying_lr=False,
    neg_sample_size: int = 10,
    num_epochs: int = 100,
    stopping_condition=no_stopping,
    lr: float = 0.1,
    scheduler_opts: Optional[Tuple[int, float]] = None,
):

    model = criterion.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if decaying_lr:
        # Linearly decaying lr (run it for half of training time)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, total_iters=int((num_epochs * len(train_loader)) / 2)
        )

    batch_metrics = []
    batch_metrics.append(evaluation_metric(model))
    batch_losses = []

    for epoch in range(num_epochs):

        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, (y, idx) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            # TODO: might be a better way to add wd
            if weight_decay > 0.0:
                # Update model gradients with weight decay grad.
                for param in model.parameters():
                    param.grad += weight_decay * param.detach().clone()

            # Take gradient step
            optimizer.step()
            if decaying_lr:
                scheduler.step()

            # Note: now logging this for every iteration (and not epoch)
            with torch.no_grad():
                loss = criterion.crit(y, None)
                batch_losses.append(loss.item())
            batch_metrics.append(evaluation_metric(model))

        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):
            print("Training converged")
            break

    # print("Finished training")

    if save_dir is not None:
        np.save(save_dir, batch_metrics)
        print("Data saved")

    return torch.tensor(batch_losses), torch.tensor(batch_metrics)


def train_ace_model(
    criterion,
    train_loader,
    validation_loader,
    save_dir,
    weight_decay=0.0,
    decaying_lr=False,
    num_training_steps: int = 1600000,
    num_warm_up_steps: int = 5000,
    lr: float = 0.1,
    scheduler_opts: Optional[Tuple[int, float]] = None,
    evaluation_freq=5000,
    validation_freq=5000,
    device=torch.device("cpu")
):

    #num_epochs = np.ceil(num_training_steps / validation_freq)

    model = criterion.get_model().to(device)
    proposal = criterion.get_proposal().to(device)

    torch.save(model.state_dict(), save_dir + "_model")
    torch.save(proposal.state_dict(), save_dir + "_proposal")
    best_loss_p = 1e6
    optimizer = torch.optim.Adam(list(model.parameters()) + list(proposal.parameters()), lr=lr)
    if scheduler_opts is not None:
        # Polynomial decaying lr
        num_steps_decay, lr_factor = scheduler_opts
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=PolynomialLr(num_steps_decay, lr,
                                                                             lr * lr_factor).decayed_learning_rate)

    losses, losses_p, losses_q = [], [], []
    val_losses, val_losses_p, val_losses_q = [], [], []

    step = 0
    while step < num_training_steps:

        for i, (y, idx) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            y = y.to(device)

            # Calculate and assign gradients

            if step < num_warm_up_steps:
                criterion.calculate_crit_grad_q(y, y.shape[0])
            else:
                criterion.calculate_crit_grad(y, y.shape[0])

            if weight_decay > 0.0:
                # Update model gradients with weight decay grad.

                for param in model.parameters():
                    if isinstance(param, torch.nn.Parameter):
                        param.grad += weight_decay * param.detach().clone()

                for param in proposal.parameters():
                    if isinstance(param, torch.nn.Parameter):
                        param.grad += weight_decay * param.detach().clone()

            # Take gradient step
            optimizer.step()

            if decaying_lr:
                scheduler.step()

            # TODO: Reset trainloader?
            # Note: now logging this for every epoch
            with torch.no_grad():
                model.eval()
                proposal.eval()

                # TODO: remove this?
                if np.mod(step + 1, evaluation_freq) == 0:
                    loss, loss_p, loss_q = get_ace_losses(train_loader, criterion, device)
                    losses.append(loss)
                    losses_p.append(loss_p)
                    losses_q.append(loss_q)

                    print("[Step {}]  Loss model: {:.3f} | Loss proposal: {:.3f} "
                          "| Loss total: {:.3f}".format(step + 1, loss_p, loss_q, loss))

                if np.mod(step + 1, validation_freq) == 0:

                    val_loss, val_loss_p, val_loss_q = get_ace_losses(validation_loader, criterion, device)
                    val_losses.append(val_loss)
                    val_losses_p.append(val_loss_p)
                    val_losses_q.append(val_loss_q)

                    print("[Step {}]  Val loss model: {:.3f} | Val loss proposal: {:.3f} "
                          "| Val loss total: {:.3f}".format(step + 1, val_loss_p, val_loss_q, val_loss))

                    if val_loss_p < best_loss_p:
                        print("New best model with validation loss {}".format(val_loss_p))
                        torch.save(model.state_dict(), save_dir + "_model")
                        torch.save(proposal.state_dict(), save_dir + "_proposal")
                        best_loss_p = val_loss_p

                # else:
                #     # TODO: it is a bit unnecessary to recalculate this
                #     loss, loss_p, loss_q = criterion.crit(y, None)
                #
                #     print("[Step {}]  Loss model: {:.3f} | Loss proposal: {:.3f} "
                #           "| Loss total: {:.3f}".format(step + 1, loss_p, loss_q, loss))

                model.train()
                proposal.train()

            step += 1

            #if step == num_training_steps:
            #    break

    if save_dir is not None:
        np.save(save_dir + "_train_loss", torch.tensor(losses))
        np.save(save_dir + "_val_loss", torch.tensor(val_losses))

        print("Data saved")

    return torch.tensor(losses), torch.tensor(val_losses)


def train_aem_model(
    criterion,
    train_loader,
    validation_loader,
    save_dir,
    decaying_lr=False,
    num_training_steps: int = 1600000,
    num_warm_up_steps: int = 5000,
    hard_warmup: bool = True,
    lr: float = 0.1,
    validation_freq=5000,
    device=torch.device("cpu")
    ):

    if not os.path.exists(save_dir + "/log"):
        os.makedirs(save_dir + "/log")

    writer = SummaryWriter(log_dir=save_dir + "/log")

    model = criterion.get_model().to(device)
    proposal = criterion.get_proposal().to(device)
    made = proposal.get_autoregressive_net().to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(list(made.parameters()) + list(model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_training_steps)

    # Training loop
    torch.save(model.state_dict(), save_dir + "/model")
    torch.save(proposal.state_dict(), save_dir + "/proposal")
    best_loss_p = 1e6
    tbar = tqdm(range(num_training_steps))
    for step in tbar:
        #print("Step: {}".format(step))
        optimizer.zero_grad()

        # training step
        y = next(train_loader).to(device)

        if step < num_warm_up_steps:
            if hard_warmup:
                criterion.calculate_crit_grad_q(y, y.shape[0])
            else:
                alpha = torch.Tensor([min(step / num_warm_up_steps, 1)])
                criterion.set_alpha(alpha)
                criterion.calculate_crit_grad(y, y.shape[0])

                if (step + 1) == num_warm_up_steps:
                    criterion.set_alpha(1.0)
        else:
            criterion.calculate_crit_grad(y, y.shape[0])

        optimizer.step()

        if decaying_lr:
            scheduler.step()  # TODO: har flyttat ner denna

        if (step + 1) % validation_freq == 0:

            model.eval()
            made.eval()
            criterion.set_training(False)

            with torch.no_grad():
                val_loss, val_loss_p, val_loss_q = get_aem_losses(validation_loader, criterion, device)

            if step >= num_warm_up_steps and val_loss_p < best_loss_p:
                print("New best model with validation loss {}".format(val_loss_p))
                torch.save(model.state_dict(), save_dir + "/model")
                torch.save(proposal.state_dict(), save_dir + "/proposal")
                best_loss_p = val_loss_p


            # s = 'Loss: {:.4f}, ' \
            #     'train: {:.4f}, ' \
            #     'val: {:.4f}, ' \
            #     'train prop: {:.4f}, ' \
            #     'val prop: {:.4f}'.format(
            #     loss.item(),
            #     mean_log_density.item(),
            #     running_val_log_density.item(),
            #     mean_log_proposal_density.item(),
            #     running_val_log_proposal_density.item()
            # )
            s = 'val loss: {:.4f}, ' \
                'val loss p: {:.4f}, ' \
                'val loss q: {:.4f}'.format(
                val_loss.item(),
                val_loss_p.item(),
                val_loss_q.item()
            )

            tbar.set_description(s)

            # write summaries
            # summaries = {
            #     'loss': loss.detach(),
            #     'log-prob-aem': mean_log_density.detach(),
            #     'log-prob-proposal': mean_log_proposal_density.detach(),
            #     'log-prob-aem-val': torch.Tensor([running_val_log_density]),
            #     'log-prob-proposal-val': torch.Tensor([running_val_log_proposal_density]),
            #     'log-normalizer': mean_log_normalizer.detach(),
            #     'learning-rate': torch.Tensor(scheduler.get_last_lr()),  # get_lr()
            # }

            summaries = {
                'log-prob-aem-val': torch.Tensor([val_loss]),
                'log-prob-model-val': torch.Tensor([val_loss_p]),
                'log-prob-proposal-val': torch.Tensor([val_loss_q]),
                'learning-rate': torch.Tensor(scheduler.get_last_lr())
            }

            for summary, value in summaries.items():
                writer.add_scalar(tag=summary, scalar_value=value, global_step=step)

            #torch.save(model.state_dict(), save_dir + "/model")
            #torch.save(proposal.state_dict(), save_dir + "/proposal")

            model.train()
            made.train()
            criterion.set_training(True)

    writer.flush()
    writer.close()
    #torch.save(model.state_dict(), save_dir + "/model")
    #torch.save(proposal.state_dict(), save_dir + "/proposal")


