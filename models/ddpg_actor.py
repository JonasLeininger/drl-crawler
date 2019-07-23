import numpy as np
import torch
import torch.nn as nn


class DDPGActor(nn.Module):
    """
    Deep Deterministic Policy Gradient Actor Neural Network
    Model-free algorithm to learn policies
    Outputs the actions in an interval [-1,1]
    """

    def __init__(self, config, hidden_units=(400, 300)):
        super(DDPGActor, self).__init__()
        self.fc1 = nn.Linear(config.state_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], config.action_dim)
        self.reset_parameters()
        self.device = config.device
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
