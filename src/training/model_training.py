"""Training loops"""
from typing import Optional, Tuple
import torch
from torch.optim.lr_scheduler import LinearLR
import numpy as np

from src.training.training_utils import no_stopping
from src.noise_distr.normal import MultivariateNormal


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


def train_model_model_proposal(
    model,
    crit_constructor,
    evaluation_metric,
    train_loader,
    save_dir,
    neg_sample_size: int = 10,
    num_epochs: int = 100,
    stopping_condition=no_stopping,
    lr: float = 0.1,
    scheduler_opts: Optional[Tuple[int, float]] = None,
):
    """Training loop for q = p_theta

    Training loop for idealised experiment where we have access to normalised p_theta
    such that we can sample from it and use q = p_theta as the proposal distribution.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    batch_metrics = []
    batch_metrics.append(evaluation_metric(model))
    batch_losses = []

    if scheduler_opts is not None:
        num_epochs_decay, lr_factor = scheduler_opts
        num_epochs_decay = (
            num_epochs_decay if num_epochs > num_epochs_decay else num_epochs
        )
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=lr_factor,
            total_iters=int((num_epochs_decay * len(train_loader))),
        )

    for epoch in range(1, num_epochs + 1):
        # print(f"Epoch {epoch}")
        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for _, (y, idx) in enumerate(train_loader, 0):
            q = MultivariateNormal(
                model.mu.detach().clone(), model.cov().clone().detach().clone()
            )
            criterion = crit_constructor(model, q, neg_sample_size)
            optimizer.zero_grad()
            with torch.no_grad():
                loss = criterion.crit(y, None)
                batch_losses.append(loss.item())
                # print(loss)
            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            # Take gradient step
            optimizer.step()
            if scheduler_opts is not None:
                scheduler.step()

            # running_loss += loss.item()
            with torch.no_grad():
                batch_metrics.append(evaluation_metric(model))
        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):
            print("Training converged")
            break
    # print("Finished training")
    return torch.tensor(batch_losses), torch.tensor(batch_metrics)


def train_model_adaptive_proposal(
    p_theta,
    q_phi,
    p_criterion,
    q_criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    neg_sample_size,
    num_epochs,
    stopping_condition=no_stopping,
    scheduler_opts: Optional[Tuple[int, float]] = None,
    lr: float = 0.1,
):
    """Training loop for adaptive proposal q_phi

    Training loop for jointly learning p_tilde_theta and q_phi.
    Where we assume that we can sample and evaluate q_phi.
    """
    p_optimizer = torch.optim.SGD(p_theta.parameters(), lr=lr)
    q_optimizer = torch.optim.SGD(q_phi.parameters(), lr=lr)
    batch_metrics = []
    batch_metrics.append(evaluation_metric(p_theta))
    batch_losses = []

    if scheduler_opts is not None:
        num_epochs_decay, lr_factor = scheduler_opts
        num_epochs_decay = (
            num_epochs_decay if num_epochs > num_epochs_decay else num_epochs
        )
        p_scheduler = LinearLR(
            p_optimizer,
            start_factor=1.0,
            end_factor=lr_factor,
            total_iters=int((num_epochs_decay * len(train_loader))),
        )

        q_scheduler = LinearLR(
            p_optimizer,
            start_factor=1.0,
            end_factor=lr_factor,
            total_iters=int((num_epochs_decay * len(train_loader))),
        )

    for epoch in range(1, num_epochs + 1):
        # print(f"Epoch {epoch}")
        old_params = torch.nn.utils.parameters_to_vector(q_phi.parameters())
        for _, (y, idx) in enumerate(train_loader, 0):

            # with torch.no_grad():
            #    p_loss = p_criterion.crit(y, None)
            #    batch_losses.append(p_loss.item())
            # Calculate and assign gradients
            p_optimizer.zero_grad()
            p_criterion.calculate_crit_grad(y, idx)
            p_optimizer.step()

            q_optimizer.zero_grad()
            q_criterion.calculate_crit_grad(y, idx)
            q_optimizer.step()

            if scheduler_opts is not None:
                p_scheduler.step()
                q_scheduler.step()
            with torch.no_grad():
                batch_metrics.append(evaluation_metric(p_theta))

        if stopping_condition(
            torch.nn.utils.parameters_to_vector(q_phi.parameters()), old_params
        ):
            print("Training converged")
            break
    # print("Finished training")
    return torch.tensor(batch_losses), torch.tensor(batch_metrics)


def train_model_pers_adaptive_proposal(
    p_theta,
    q_phi,
    p_criterion,
    q_criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    neg_sample_size,
    num_epochs,
    stopping_condition=no_stopping,
    scheduler_opts: Optional[Tuple[int, float]] = None,
    lr: float = 0.1,
):
    """Training loop for adaptive proposal q_phi

    Training loop for jointly learning p_tilde_theta and q_phi.
    Where we assume that we can sample and evaluate q_phi.
    """
    p_optimizer = torch.optim.SGD(p_theta.parameters(), lr=lr)
    q_optimizer = torch.optim.SGD(q_phi.parameters(), lr=lr)
    batch_metrics = []
    batch_metrics.append(evaluation_metric(p_theta))
    batch_losses = []

    if scheduler_opts is not None:
        num_epochs_decay, lr_factor = scheduler_opts
        num_epochs_decay = (
            num_epochs_decay if num_epochs > num_epochs_decay else num_epochs
        )
        p_scheduler = LinearLR(
            p_optimizer,
            start_factor=1.0,
            end_factor=lr_factor,
            total_iters=int((num_epochs_decay * len(train_loader))),
        )

        q_scheduler = LinearLR(
            p_optimizer,
            start_factor=1.0,
            end_factor=lr_factor,
            total_iters=int((num_epochs_decay * len(train_loader))),
        )

    # Init. a batch of persistent ys with a sample from data.
    for epoch in range(1, num_epochs + 1):
        # print(f"Epoch {epoch}")
        old_params = torch.nn.utils.parameters_to_vector(q_phi.parameters())
        for _, (y, idx) in enumerate(train_loader, 0):

            # with torch.no_grad():
            #    p_loss = p_criterion.crit(y, None)
            #    batch_losses.append(p_loss.item())
            # Calculate and assign gradients
            p_optimizer.zero_grad()
            p_criterion.calculate_crit_grad(y, idx)
            p_optimizer.step()

            q_optimizer.zero_grad()
            q_criterion.calculate_crit_grad(y, idx)
            q_optimizer.step()

            if scheduler_opts is not None:
                p_scheduler.step()
                q_scheduler.step()
            with torch.no_grad():
                batch_metrics.append(evaluation_metric(p_theta))

        if stopping_condition(
            torch.nn.utils.parameters_to_vector(q_phi.parameters()), old_params
        ):
            print("Training converged")
            break
    # print("Finished training")
    return torch.tensor(batch_losses), torch.tensor(batch_metrics)
