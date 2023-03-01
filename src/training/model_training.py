from typing import Optional, Tuple
import torch
import numpy as np

from src.training.training_utils import no_stopping, PolynomialLr

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
    device=torch.device("cpu")
):

    model = criterion.get_model().to(device)
    proposal = criterion.get_proposal().to(device)

    optimizer = torch.optim.SGD(list(model.parameters()) + list(proposal.parameters()), lr=lr)

    if scheduler_opts is not None:
        # Polynomial decaying lr
        # What happens after num_steps_decay?
        num_steps_decay, lr_factor = scheduler_opts
       # scheduler = torch.optim.lr_scheduler.PolynomialLR(
        #    optimizer, total_iters=num_steps_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=PolynomialLr(num_steps_decay, lr,
                                                                             lr * lr_factor).decayed_learning_rate)

    batch_metrics = []
    batch_metrics.append(evaluation_metric(model))
    batch_losses = []

    for epoch in range(num_epochs):

        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, (y, idx) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            y = y.to(device)

            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            if weight_decay > 0.0:
                # Update model gradients with weight decay grad.

                # TODO: does this cover all parameters? (i.e. we get gradient of e.g activation/dropout layer?)
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
