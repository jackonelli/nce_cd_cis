import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_model import BaseModel


class Ebm(BaseModel):
    def __init__(self, input_dim=1, hidden_dim=10):
        super().__init__()

        self.fc1_y = nn.Linear(input_dim, hidden_dim)
        self.fc2_y = nn.Linear(hidden_dim, hidden_dim)

        self.fc1_xy = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2_xy = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_xy = nn.Linear(hidden_dim, 1)

    def log_prob(self, x, y):
        """Compute log of unnorm prob: log p_tilde(y)

        This corresponds to the negative energy, - E_param(x, y)
        """
        # (x_feature has shape: (batch_size, hidden_dim))
        # (y has shape (batch_size, num_samples)) (num_sampes==1 when running on (x_i, y_i))

        if y.dim() == 1:
            y = y.view(-1, 1)

        batch_size, num_samples = y.shape

        # Replicate
        x = x.view(batch_size, 1, -1).expand(
            -1, num_samples, -1
        )  # (shape: (batch_size, num_samples, hidden_dim))

        # resize to batch dimension
        x = x.reshape(
            batch_size * num_samples, -1
        )  # (shape: (batch_size*num_samples, hidden_dim))
        y = y.reshape(
            batch_size * num_samples, -1
        )  # (shape: (batch_size*num_samples, 1))

        y_feature = F.relu(
            self.fc1_y(y)
        )  # (shape: (batch_size*num_samples, hidden_dim))
        y_feature = F.relu(
            self.fc2_y(y_feature)
        )  # (shape: (batch_size*num_samples, hidden_dim))

        xy_feature = torch.cat(
            [x, y_feature], 1
        )  # (shape: (batch_size*num_samples, 2*hidden_dim))

        xy_feature = F.relu(
            self.fc1_xy(xy_feature)
        )  # (shape: (batch_size*num_samples, hidden_dim))
        xy_feature = (
            F.relu(self.fc2_xy(xy_feature)) + xy_feature
        )  # (shape: (batch_size*num_samples, hidden_dim))
        energy = self.fc3_xy(xy_feature)  # (shape: (batch_size*num_samples, 1))

        energy = energy.view(
            batch_size, num_samples
        )  # (shape: (batch_size, num_samples))

        return -energy
