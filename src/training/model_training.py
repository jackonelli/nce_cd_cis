import torch
import numpy as np


# TODO: evaluate convergence and stop if converged?
def train_model(criterion, evaluation_metric, train_loader, save_dir, neg_sample_size=10, num_epochs=100):

    model = criterion._unnorm_distr
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    running_loss_it = np.zeros(num_epochs)

    metric = np.zeros((num_epochs,))
    for epoch in range(num_epochs):

        running_loss = 0.0
        for i, y in enumerate(train_loader, 0):

            optimizer.zero_grad()

            y_samples = criterion._noise_distr.sample(neg_sample_size, y)
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

    print('Finished training')
    np.save(save_dir, metric)
    print("Data saved")

    return metric[-1]


class EuclideanPrecisionMetric:

    def __init__(self, true_precision):
        self.true_precison = true_precision

    def metric(self, model):
        return torch.mean((torch.exp(model.log_precision) - self.true_precison) ** 2)