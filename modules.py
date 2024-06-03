import torch
from torch import nn
import numpy as np

class OptimalModule(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.aux_states_dim = 5+2+1+1
        self.hidden_dim = 32
        self.time_dim = 1
        self.ref_dim = 10001
        self.q_dim = 32
        self.k_dim = 32
        self.v_dim = 32
        self.state_fc_block = nn.Sequential(
            nn.Linear(self.aux_states_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.hidden_dim),
            nn.LeakyReLU()
        )
        self.ref_fc_block = nn.Sequential(
            nn.Linear(self.ref_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.time_fc_block = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.control_fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim+self.v_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 3)
        )
        self.q_block = nn.Sequential(
            nn.Linear(self.hidden_dim, self.q_dim),
            nn.LeakyReLU(),
            nn.Linear(self.q_dim, self.q_dim)
        )
        self.k_block = nn.Sequential(
            nn.Linear(1, self.k_dim),
            nn.LeakyReLU(),
            nn.Linear(self.k_dim, self.k_dim)
        )
        self.v_block = nn.Sequential(
            nn.Linear(1, self.v_dim),
            nn.LeakyReLU(),
            nn.Linear(self.v_dim, self.v_dim)
        )


    def forward(self, input):
        aux_states = input[:, 0:self.aux_states_dim]
        time = input[:, self.aux_states_dim:self.aux_states_dim+self.time_dim]
        h_r = input[:, self.aux_states_dim+self.time_dim:]
        h_r = torch.unsqueeze(h_r, -1)
        hidden = self.state_fc_block(aux_states)  # n_batch*d_hidden

        # Attention flow
        query = self.q_block(hidden)  # n_batch*d_q
        query = torch.unsqueeze(query, dim=1)  # n_batch*1*d_q
        key = self.k_block(h_r)  # n_batch*step_num*d_k
        value = self.v_block(h_r)  # n_batch*step_num*d_v
        # d_q == d_k, then
        score = torch.matmul(query, torch.transpose(key, 1, 2))/np.sqrt(self.q_dim)  # n_batch*1*step_num
        score = torch.softmax(score, dim=2)  # n_batch*1*step_num
        att_out = torch.matmul(score, value)  # n_batch*1*d_v
        att_out = torch.squeeze(att_out, dim=1)  # n_batch*d_v

        # r_info = self.ref_fc_block(h_r)
        # time_info = self.time_fc_block(time)
        output = self.control_fc_block(torch.cat((hidden, att_out), dim=1))
        return output
    




