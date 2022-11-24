import torch
import numpy as np

from src.training.training_utils import no_stopping


def train_model(criterion, evaluation_metric, train_loader, save_dir, neg_sample_size=10, num_epochs=100,
                stopping_condition=no_stopping):

    model = criterion.get_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    running_loss_it = np.zeros(num_epochs)

    metric = np.zeros((num_epochs,))
    for epoch in range(num_epochs):

        running_loss = 0.0
        old_params = torch.nn.utils.parameters_to_vector(model.parameters())
        for i, y in enumerate(train_loader, 0):

            optimizer.zero_grad()

            y_samples = criterion.sample_noise(neg_sample_size * y.size(0), y)
            loss = criterion.crit(y, y_samples)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print statistics
        print('[%d] loss: %.3f' %
              (epoch + 1, running_loss))
        running_loss_it[epoch] = running_loss

        metric[epoch] = evaluation_metric(model)

        print('[%d] evaluation metric: %.3f' %
              (epoch + 1, metric[epoch]))

        if stopping_condition(torch.nn.utils.parameters_to_vector(model.parameters()), old_params):
            print("Training converged")
            break

    print('Finished training')
    np.save(save_dir, metric)
    print("Data saved")

    return metric[-1]



