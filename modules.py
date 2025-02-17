import torch
from torch import nn
import numpy as np
import config_opc

class OptimalModule(nn.Module):

    def __init__(self, *args):
        super().__init__()
        self.dt = config_opc.PARA_DT
        self.aux_states_dim = config_opc.AUX_STATES_DIM
        self.hidden_dim = config_opc.HIDDEN_DIM
        self.time_dim = config_opc.TIME_DIM
        self.ref_dim = config_opc.REF_DIM
        self.q_dim = config_opc.Q_DIM
        self.k_dim = config_opc.K_DIM
        self.v_dim = config_opc.V_DIM
        self.state_fc_block = nn.Sequential(
            nn.Linear(self.aux_states_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.hidden_dim),
            nn.LeakyReLU()
        )
        self.skip_layer = nn.Linear(self.aux_states_dim, self.hidden_dim)

        self.ref_fc_block = nn.Sequential(
            nn.Linear(self.ref_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.time_fc_block = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.control_fc_block = nn.Sequential(
            nn.Linear(self.hidden_dim+self.v_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 3)
        )
        # self.q_block = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.q_dim),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(self.q_dim, self.q_dim),
        #     # nn.LeakyReLU(),
        #     nn.Linear(self.q_dim, self.q_dim)
        # )
        # self.k_block = nn.Sequential(
        #     nn.Linear(1, self.k_dim),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(self.k_dim, self.k_dim),
        #     # nn.LeakyReLU(),
        #     nn.Linear(self.k_dim, self.k_dim)
        # )
        # self.v_block = nn.Sequential(
        #     nn.Linear(1, self.v_dim),
        #     # nn.LeakyReLU(),
        #     # nn.Linear(self.v_dim, self.v_dim),
        #     # nn.LeakyReLU(),
        #     nn.Linear(self.v_dim, self.v_dim)
        # )
        self.control_skip_layer = nn.Sequential(
            nn.Linear(self.aux_states_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 3)
        )


    def forward(self, input, mask_mat):
        aux_states = input[:, 0:self.aux_states_dim]
        time = input[:, self.aux_states_dim:self.aux_states_dim+self.time_dim]
        h_r = input[:, self.aux_states_dim+self.time_dim:]
        h_r = torch.unsqueeze(h_r, -1)
        hidden = self.state_fc_block(aux_states)
        # hidden = self.state_fc_block(aux_states) + self.skip_layer(aux_states)  # n_batch*d_hidden

        # Attention flow
        # query = self.q_block(hidden)  # n_batch*d_q
        # query = torch.unsqueeze(query, dim=1)  # n_batch*1*d_q
        # key = self.k_block(h_r)  # n_batch*step_num*d_k
        # value = self.v_block(h_r)  # n_batch*step_num*d_v
        # # d_q == d_k, then
        # score = torch.matmul(query, torch.transpose(key, 1, 2))/np.sqrt(self.q_dim)  # n_batch*1*step_num
        # # time_index = int(time/self.dt)
        # # score[0:time_index] = -1e8  # mask
        # score = score + mask_mat
        # score = torch.softmax(score, dim=2)  # n_batch*1*step_num
        # # torch.nn.MultiheadAttention()
        # att_out = torch.matmul(score, value)  # n_batch*1*d_v
        # att_out = torch.squeeze(att_out, dim=1)  # n_batch*d_v

        # r_info = self.ref_fc_block(h_r)
        # time_info = self.time_fc_block(time)
        # output = self.control_fc_block(torch.cat((hidden, att_out), dim=1))
        output = self.control_fc_block(hidden)
        # output = self.control_fc_block(torch.cat((hidden, att_out), dim=1)) + self.control_skip_layer(aux_states)
        return output
    




