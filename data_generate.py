import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime

import config_opc
import dynamics2 as dyn
import simulate as simu
import optimal as opt
import plot_utils as pu

class OptimalDataset(Dataset):
    def __init__(self):
        # super().__init__()
        self.x_all = None
        self.y_all = None
        self.z_all = None
        self.u_all = None
        self.h_r = None
        self.time_steps = None
        self.have_item = False
        self.step_num = 0

        # # Normalization
        # self.x_mean = np.mean(x_all_simu, axis=0)
        # self.x_std = np.std(x_all_simu, axis=0)
        # self.x_all = (x_all_simu-self.x_mean)/self.x_std
        # self.y_mean = np.mean(y_all_simu, axis=0)
        # self.y_std = np.std(y_all_simu, axis=0)
        # self.y_all = (y_all_simu-self.y_mean)/self.y_std
        # self.z_mean = np.mean(z_all_simu, axis=0)
        # self.z_std = np.std(z_all_simu, axis=0)
        # self.z_all = (z_all_simu-self.z_mean)/self.z_std
        # self.u_mean = np.mean(u_all_simu, axis=0)
        # self.u_std = np.std(u_all_simu, axis=0)
        # # self.u_mean = 0
        # # self.u_std = 1
        # self.u_all = (u_all_simu-self.u_mean)/self.u_std
        # self.h_r_mean = np.mean(tra_ref["h_r_seq"])
        # self.h_r_std = np.std(tra_ref["h_r_seq"])
        # self.h_r = (tra_ref["h_r_seq"] - self.h_r_mean)/self.h_r_std
        # self.time_steps = tra_ref["time_steps"]/config_opc.PARA_TF
    
    def __len__(self):
        return self.u_all.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        aux_states = np.concatenate((self.x_all[idx, :], self.y_all[idx, :], self.z_all[idx, :], self.h_r[idx][np.newaxis]), axis=0)  # Shape: (5+2+1+1, 1)
        control_value = self.u_all[idx, :]
        t = self.time_steps[idx][np.newaxis]
        no = idx//self.step_num
        sample = {"input": np.concatenate((aux_states, t, self.h_r[no*self.step_num:(no+1)*self.step_num])).astype(np.float32), "output": control_value.astype(np.float32)}
        return sample
    
    def append_item(self, x_all_simu, y_all_simu, z_all_simu, u_all_simu, tra_ref):
        if self.have_item:
            self.x_all = np.concatenate((self.x_all, x_all_simu), axis=0)
            self.y_all = np.concatenate((self.y_all, y_all_simu), axis=0)
            self.z_all = np.concatenate((self.z_all, z_all_simu), axis=0)
            self.u_all = np.concatenate((self.u_all, u_all_simu), axis=0)
            self.h_r = np.concatenate((self.h_r, tra_ref["h_r_seq"]), axis=0)
            self.time_steps = np.concatenate((self.time_steps, tra_ref["time_steps"]), axis=0)
        else:
            self.x_all = x_all_simu
            self.y_all = y_all_simu
            self.z_all = z_all_simu
            self.u_all = u_all_simu
            self.h_r = tra_ref["h_r_seq"]
            self.time_steps = tra_ref["time_steps"]
            self.step_num = self.u_all.shape[0]
            print(f"Step num: {self.step_num}")
            self.have_item = True

    
    def normalization(self):
        self.x_mean = np.mean(self.x_all, axis=0)
        self.x_std = np.std(self.x_all, axis=0)
        self.x_all = (self.x_all-self.x_mean)/self.x_std
        self.y_mean = np.mean(self.y_all, axis=0)
        self.y_std = np.std(self.y_all, axis=0)
        self.y_all = (self.y_all-self.y_mean)/self.y_std
        self.z_mean = np.mean(self.z_all, axis=0)
        self.z_std = np.std(self.z_all, axis=0)
        self.z_all = (self.z_all-self.z_mean)/self.z_std
        self.u_mean = np.mean(self.u_all, axis=0)
        self.u_std = np.std(self.u_all, axis=0)
        # self.u_mean = 0
        # self.u_std = 1
        self.u_all = (self.u_all-self.u_mean)/self.u_std
        self.h_r_mean = np.mean(self.h_r)
        self.h_r_std = np.std(self.h_r)
        self.h_r = (self.h_r - self.h_r_mean)/self.h_r_std
        self.time_steps = self.time_steps/config_opc.PARA_TF
    

def save_statistics(dataset, time_stamp):
    np.savez(f'data/opt_stats_{time_stamp}.npz', 
             x_mean=dataset.x_mean, x_std=dataset.x_std,
             y_mean=dataset.y_mean, y_std=dataset.y_std,
             z_mean=dataset.z_mean, z_std=dataset.z_std,
             u_mean=dataset.u_mean, u_std=dataset.u_std,
             h_r_mean=dataset.h_r_mean, h_r_std=dataset.h_r_std)
    return

if __name__ == "__main__":

    # switch_time = 0.5
    # high_height = 350
    # low_height = 250
    # tra_ref = simu.generate_ref_trajectory_varying(switch_time=switch_time, high_height=high_height, low_height=low_height)
    # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    # t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
    # x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u)
    # dataset = OptimalDataset(x_all_simu, y_all_simu, z_all_simu, u_all_simu, tra_ref)
    # dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     print(sample_batched["input"][0, 0:8])
    #     print(sample_batched["output"][0, :])
    # torch.save(dataset, 'data/opt_data.pt')
    # save_statistics(dataset)
    # pass
    
    time_current = datetime.now().strftime('%m-%d-%H%M')
    dataset = OptimalDataset()
    err_thr = 3000
    switch_time = 0.5
    high_heights = np.arange(start=350, stop=375, step=5)
    low_heights = np.arange(start=250, stop=225, step=-5)
    high_heights = np.arange(start=350, stop=355, step=5)
    low_heights = np.arange(start=250, stop=245, step=-5)
    pairs = []
    for h in high_heights:
        for l in low_heights:
            tra_ref = simu.generate_ref_trajectory_varying(switch_time=switch_time, high_height=h, low_height=l)
            # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
            x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_scaled(x0=config_opc.PARA_X0,trajectory_ref=tra_ref, morphing_disabled=None, fun_obj=opt.function_objective_both, maxiter=500)
            t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
            x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u)
            if j_f_simu[0] < err_thr:
                pairs.append({'high_height':h, 'low_height':l, 'error':j_f_simu[0], 'if_add':True})
                dataset.append_item(x_all_simu, y_all_simu, z_all_simu, u_all_simu, tra_ref)
            else:
                pairs.append({'high_height':h, 'low_height':l, 'error':j_f_simu[0], 'if_add':False})
    
    for pair in pairs:
        print(pair)

    dataset.normalization()
        
    torch.save(dataset, f'data/opt_data_{time_current}.pt')
    save_statistics(dataset, time_current)
    pass