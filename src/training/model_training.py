import torch
import numpy as np

from src.training.training_utils import no_stopping
from src.noise_distr.normal import MultivariateNormal
from src.nce.rank import NceRankCrit


def train_model_model_proposal(
    model,
    criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    neg_sample_size: int = 10,
    num_epochs: int = 100,
    stopping_condition=no_stopping,
    lr: float = 0.1,
):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    metric = []
    for epoch in range(1, num_epochs + 1):
        # print(f"Epoch {epoch}")
        running_loss = 0.0
        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for _, (y, idx) in enumerate(train_loader, 0):
            q = MultivariateNormal(
                model.mu.detach().clone(), model.cov().clone().detach().clone()
            )
            criterion = NceRankCrit(model, q, J)
            optimizer.zero_grad()
            with torch.no_grad():
                loss = criterion.crit(y, None)
                # print(loss)
            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            # Take gradient step
            optimizer.step()

            # running_loss += loss.item()
            metric.append(evaluation_metric(model).detach().numpy())
        if stopping_condition(
            torch.nn.utils.parameters_to_vector(model.parameters()), old_params
        ):


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

    metric = []
    for epoch in range(num_epochs):

        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, (y, idx) in enumerate(train_loader, 0):

            optimizer.zero_grad()

            # Calculate and assign gradients
            criterion.calculate_crit_grad(y, idx)

            # Take gradient step
            optimizer.step()

            # Note: now logging this for every iteration (and not epoch)
            metric.append(evaluation_metric(model).detach().numpy())

        # print('[%d] evaluation metric: %.3f' %
        #       (epoch + 1, metric[-1]))

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
