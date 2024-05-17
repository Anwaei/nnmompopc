import torch
from torch import nn


class OptimalModule(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.aux_states_dim = 5+2+1+1
        self.ref_dim = 10001
        self.state_fc_block = nn.Sequential(
            nn.Linear(self.aux_states_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.ref_fc_block = nn.Sequential(
            nn.Linear(self.ref_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.control_fc_block = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )

    def forward(self, input):
        aux_states = input[:, 0:self.aux_states_dim]
        h_r = input[:, self.aux_states_dim:]
        hidden = self.state_fc_block(aux_states)
        r_info = self.ref_fc_block(h_r)
        output = self.control_fc_block(torch.cat((hidden, r_info), dim=1))
        return output
    




