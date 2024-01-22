import os
from typing import Optional, Tuple
import torch
import numpy as np

from src.training.training_utils import no_stopping, PolynomialLr
from src.experiments.aem_exp_utils import get_aem_losses

from tensorboardX import SummaryWriter
from tqdm import tqdm


def train_model(
    criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    num_epochs=100,
    weight_decay=0.0,
    lr=0.1,
    decaying_lr=False,
    lr_factor=0.1,
    num_epochs_decay=100,
    stopping_condition=no_stopping,
    Adam=False,
    device=torch.device("cpu")
):

    model = criterion.get_model().to(device)

    if Adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if decaying_lr:
        # Linearly decaying lr (run it for half of training time)
        num_epochs_decay = num_epochs_decay if num_epochs > num_epochs_decay else num_epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_factor,
                                                      total_iters=int((num_epochs_decay * len(train_loader))))

    metric = []
    for epoch in range(num_epochs):

        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, (y, idx) in enumerate(train_loader, 0):

            if type(y) is tuple or type(y) is list:
                y = (y[0].to(device), y[1].to(device))
            else:
                y = y.to(device)

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
            metric.append(evaluation_metric(model).detach().numpy())

        if np.mod(epoch + 1, 10) == 0:
            print('[%d] evaluation metric: %.3f' % (epoch + 1, metric[-1]))
            torch.save(model.state_dict(), "res/model_" + str(epoch + 1))

        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):
            print("Training converged")
            break

    print("Finished training")

    metric = np.array(metric)
    if save_dir is not None:
        np.save(save_dir, metric)
        print("Data saved")

    return metric[-1]


def train_aem_model(
    criterion,
    train_loader,
    validation_loader,
    save_dir,
    decaying_lr=False,
    num_training_steps: int = 1600000,
    num_warm_up_steps: int = 5000,
    num_training_steps_q = None,
    hard_warmup: bool = True,
    lr: float = 0.1,
    validation_freq=5000,
    device=torch.device("cpu"),
    save_final=True
    ):
    
    if num_training_steps_q is None:
        num_training_steps_q = num_training_steps
    else:
        assert num_training_steps_q >= num_warm_up_steps, "Number of warm up steps larger than total training steps for q" 

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
    best_loss_q = 1e6
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
                    
        elif step >= num_training_steps_q:
            criterion.calculate_crit_grad_p(y, y.shape[0])
        else:
            criterion.calculate_crit_grad(y, y.shape[0])

        optimizer.step()

        if decaying_lr:
            scheduler.step()  # TODO: moved to here

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
                
            #elif step < num_warm_up_steps and val_loss_q < best_loss_q:
            if val_loss_q < best_loss_q:
                print("New best proposal with validation loss {}".format(val_loss_q))
                torch.save(proposal.state_dict(), save_dir + "/proposal_best")
                best_loss_q = val_loss_q


  
            s = 'val loss: {:.4f}, ' \
                'val loss p: {:.4f}, ' \
                'val loss q: {:.4f}'.format(
                val_loss.item(),
                val_loss_p.item(),
                val_loss_q.item()
            )

            tbar.set_description(s)

            summaries = {
                'log-prob-aem-val': torch.Tensor([val_loss]),
                'log-prob-model-val': torch.Tensor([val_loss_p]),
                'log-prob-proposal-val': torch.Tensor([val_loss_q]),
                'learning-rate': torch.Tensor(scheduler.get_last_lr())
            }

            for summary, value in summaries.items():
                writer.add_scalar(tag=summary, scalar_value=value, global_step=step)


            model.train()
            made.train()
            criterion.set_training(True)

    writer.flush()
    writer.close()
    
    if save_final:
        torch.save(model.state_dict(), save_dir + "/model_final")
        torch.save(proposal.state_dict(), save_dir + "/proposal_final")

