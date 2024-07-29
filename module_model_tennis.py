import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(data_layer):
    data_fan_in = data_layer.weight.data.size()[0]
    data_lim = 1. / np.sqrt(data_fan_in)
    return -data_lim, data_lim

class ModuleCritic(nn.Module):
    def __init__(self, data_state_size, data_action_size, data_seed):
        super(ModuleCritic, self).__init__()
        self.data_seed = torch.manual_seed(data_seed)
        self.data_bn1 = nn.BatchNorm1d(data_state_size)
        self.data_fcs1 = nn.Linear(data_state_size, 128)
        self.data_fc2 = nn.Linear(128+2*data_action_size, 128)
        self.data_fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def forward(self, data_state, data_action, data_action_other_player):
        data_xs = self.data_bn1(data_state)
        data_xs = F.leaky_relu(self.data_fcs1(data_xs))
        data_x = torch.cat((data_xs, data_action, data_action_other_player), dim=1)
        data_x = F.leaky_relu(self.data_fc2(data_x))
        return self.data_fc3(data_x)

    def reset_parameters(self):
        self.data_fcs1.weight.data.uniform_(*hidden_init(self.data_fcs1))
        self.data_fc2.weight.data.uniform_(*hidden_init(self.data_fc2))
        self.data_fc3.weight.data.uniform_(-3e-3, 3e-3)

class ModuleActor(nn.Module):
    def __init__(self, data_state_size, data_action_size, data_seed):
        super(ModuleActor, self).__init__()
        self.data_seed = torch.manual_seed(data_seed)
        self.data_bn1 = nn.BatchNorm1d(data_state_size)
        self.data_fc1 = nn.Linear(data_state_size, 128)
        self.data_fc2 = nn.Linear(128, 128)
        self.data_fc3 = nn.Linear(128, data_action_size)
        self.reset_parameters()

    def forward(self, data_state):
        data_x = self.data_bn1(data_state)
        data_x = F.leaky_relu(self.data_fc1(data_x))
        data_x = F.leaky_relu(self.data_fc2(data_x))
        return torch.tanh(self.data_fc3(data_x))

    def reset_parameters(self):
        self.data_fc1.weight.data.uniform_(*hidden_init(self.data_fc1))
        self.data_fc2.weight.data.uniform_(*hidden_init(self.data_fc2))
        self.data_fc3.weight.data.uniform_(-3e-3, 3e-3)

