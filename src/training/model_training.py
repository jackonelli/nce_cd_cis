"""Training loops"""
import torch
import numpy as np

from src.training.training_utils import no_stopping
from src.noise_distr.normal import MultivariateNormal


def train_model(
    criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    neg_sample_size: int = 10,
    num_epochs: int = 100,
    stopping_condition=no_stopping,
    lr: float = 0.1,
):

    model = criterion.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    batch_metrics = []
    batch_metrics.append(evaluation_metric(model))
    batch_losses = []
    for epoch in range(num_epochs):

        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, (y, idx) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            # Take gradient step
            optimizer.step()

            # Note: now logging this for every iteration (and not epoch)
            with torch.no_grad():
                loss = criterion.crit(y, None)
                batch_losses.append(loss.item())
            batch_metrics.append(evaluation_metric(model))

        # print('[%d] evaluation metric: %.3f' %
        #       (epoch + 1, metric[-1]))

        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):
            print("Training converged")
            break

    print("Finished training")

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
):
    """Training loop for q = p_theta

    Training loop for idealised experiment where we have access to normalised p_theta
    such that we can sample from it and use q = p_theta as the proposal distribution.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    batch_metrics = []
    batch_metrics.append(evaluation_metric(model))
    batch_losses = []
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

            # running_loss += loss.item()
            batch_metrics.append(evaluation_metric(model))
        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):
            print("Training converged")
            break
    return torch.tensor(batch_losses), torch.tensor(batch_metrics)
