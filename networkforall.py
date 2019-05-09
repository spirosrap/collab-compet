import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc_units=512, fc_units1 = 256,fc_units2=256,fc_units3=64, actor=False):
        super(Actor, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.bn = nn.BatchNorm1d(state_size)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0)
        self.fc1 = nn.Linear(state_size,fc_units)
        self.fc2 = nn.Linear(fc_units,fc_units1)
        self.fc3 = nn.Linear(fc_units1,fc_units2)
        self.fc4 = nn.Linear(fc_units2,fc_units3)
        self.fc5 = nn.Linear(fc_units3,action_size)

        # Output of the critic
        self.outcritic = nn.Linear(fc_units3,1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc5.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, state):
        # return a vector of the force
        x = F.relu(self.fc1(self.bn(state.contiguous())))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return torch.tanh(self.fc5(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=0, fc_units=512, fc_units1 = 256,fc_units2=256,fc_units3=64, actor=False):
        super(Critic, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.bn = nn.BatchNorm1d(state_size*2)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(0)

        self.fc1 = nn.Linear((state_size+action_size)*2,fc_units)
        self.fc2 = nn.Linear(fc_units,fc_units1)
        self.fc3 = nn.Linear(fc_units1,fc_units2)
        self.fc4 = nn.Linear(fc_units2,fc_units3)
        self.fc5 = nn.Linear(fc_units3,action_size)

        # Output of the critic
        self.outcritic = nn.Linear(fc_units3,1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc5))
        self.fc5.weight.data.uniform_(-1e-3, 1e-3)
        self.outcritic.weight.data.uniform_(-1e-3, 1e-3)
    def forward(self, state,action):
        x = torch.cat((self.bn(state.contiguous()), action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.outcritic(x)

        return x
