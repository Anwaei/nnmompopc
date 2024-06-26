import torch
import config_opc

def cal_mask_mat(x):
    # x: n_batch * n_input
    t = x[:, config_opc.AUX_STATES_DIM:config_opc.AUX_STATES_DIM+config_opc.TIME_DIM]
    step_num = config_opc.REF_DIM
    n_batch = x.shape[0]
    mask_mat = torch.zeros((n_batch, 1, step_num))
    for i in range(n_batch):
        for j in range(step_num):
            if j < t[i]:
                mask_mat[i, j] = -1e8
            else:
                break
    return mask_mat