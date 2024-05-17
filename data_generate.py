import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config_opc
import dynamics2 as dyn
import simulate as simu
import optimal as opt
import plot_utils as pu

class OptimalDataset(Dataset):
    def __init__(self, x_all_simu, y_all_simu, z_all_simu, u_all_simu, tra_ref):
        # super().__init__()
        self.x_all = x_all_simu
        self.y_all = y_all_simu
        self.z_all = z_all_simu
        self.u_all = u_all_simu
        self.h_r = tra_ref["h_r_seq"]
    
    def __len__(self):
        return self.u_all.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aux_states = np.concatenate((self.x_all[idx, :], self.y_all[idx, :], self.z_all[idx, :], self.h_r[idx][np.newaxis]), axis=0)  # Shape: (5+2+1+1, 1)
        control_value = self.u_all[idx, :]
        sample = {"input": np.concatenate((aux_states, self.h_r)).astype(np.float32), "output": control_value.astype(np.float32)}
        return sample


if __name__ == "__main__":

    switch_time = 0.5
    high_height = 350
    low_height = 250
    tra_ref = simu.generate_ref_trajectory_varying(switch_time=switch_time, high_height=high_height, low_height=low_height)
    x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
    x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u)
    dataset = OptimalDataset(x_all_simu, y_all_simu, z_all_simu, u_all_simu, tra_ref)
    # dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     print(sample_batched["input"][0, 0:8])
    #     print(sample_batched["output"][0, :])
    torch.save(dataset, 'data/opt_data.pt')
    pass
    
