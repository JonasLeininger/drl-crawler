import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGCritic(nn.Module):
    """
    Deep Deterministic Policy Gradient Critic Neural Network
    Model-free algorithm to learn policies
    Outputs the Q-Value for given states and actions
    """

    def __init__(self, config, actions_dim=20, hidden_units=(400, 300)):
        super(DDPGCritic, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + actions_dim, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        self.reset_parameters()
        self.device = config.device
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        mu = torch.cat((x, action), dim=1)
        mu = torch.relu(mu)
        mu = self.fc2(mu)
        mu = torch.relu(mu)
        mu = self.fc3(mu)
        mu = torch.relu(mu)
        return mu
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
