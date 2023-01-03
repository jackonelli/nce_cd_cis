import torch
import numpy as np

from src.training.training_utils import no_stopping


def train_model(
    criterion,
    evaluation_metric,
    train_loader,
    save_dir,
    num_epochs=100,
    weight_decay=0.0,
    lr=0.1,
    decaying_lr=False,
    stopping_condition=no_stopping,
):

    model = criterion.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if decaying_lr:
        # Linearly decaying lr (run it for half of training time)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=int((num_epochs * len(train_loader)) / 2))  

    metric = []
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

