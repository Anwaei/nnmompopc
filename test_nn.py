import time, os
import torch
import numpy as np
from modules import OptimalModule
import simulate as simu
import optimal as opt
import config_opc
import plot_utils as pu
from calculate_utils import cal_mask_mat

if __name__ == '__main__':
    net_path = 'model/net_02-23-2329/epoch_996.pth'
    net = OptimalModule()
    net.load_state_dict(torch.load(net_path))
    net.eval()
    opt_stats = np.load('data/opt_stats_02-22-2343.npz')
    x_mean = opt_stats['x_mean']
    x_std = opt_stats['x_std']
    y_mean = opt_stats['y_mean']
    y_std = opt_stats['y_std']
    z_mean = opt_stats['z_mean']
    z_std = opt_stats['z_std']
    u_mean = opt_stats['u_mean']
    u_std = opt_stats['u_std']
    h_r_mean = opt_stats['h_r_mean']
    h_r_std = opt_stats['h_r_std']

    switch_time = 0.5
    high_height = 350
    low_height = 250
    tra_ref = simu.generate_ref_trajectory_varying(switch_time=switch_time, high_height=high_height, low_height=low_height)
    # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    # t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
    # x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u)
    # h_r_seq = tra_ref['h_r_seq']
    # time_steps = tra_ref['time_steps']

    file_folder_m_b = "pics/results_open_20250210-0038_14"
    with np.load(f'{file_folder_m_b}\\data_morphing_both.npz') as data_morphing_both:
                    keys = ['x_m_b', 'y_m_b', 'z_m_b', 'u_m_b', 'j_f_m_b', 'aero_info_m_b']
                    results_m_b = [data_morphing_both[key] for key in keys]
    
    x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = results_m_b
    h_r_seq = tra_ref['h_r_seq']
    time_steps = tra_ref['time_steps']

    u_all_pred = np.zeros((config_opc.PARA_STEP_NUM, config_opc.PARA_NU_AUXILIARY))
    with torch.no_grad():
        net_input_ref = (h_r_seq-h_r_mean)/h_r_std
        for k in range(config_opc.PARA_STEP_NUM):
            x_normalized = (x_all_simu[k, :]-x_mean)/x_std
            y_normalized = (y_all_simu[k, :]-y_mean)/y_std
            z_normalized = (z_all_simu[k, :]-z_mean)/z_std
            h_r_normalized = net_input_ref[k]
            
            net_input_state = np.concatenate((x_normalized, y_normalized, z_normalized, h_r_normalized[np.newaxis]), axis=0)
            net_input_time = time_steps[k][np.newaxis]/config_opc.PARA_TF
            net_input = np.concatenate((net_input_state, net_input_time, net_input_ref)).astype(np.float32)
            net_input = torch.from_numpy(np.expand_dims(net_input, axis=0))
            mask_mat = cal_mask_mat(net_input)
            u_predict = net(net_input, mask_mat)
            u_normalized = u_predict.detach().numpy().astype(np.float64)            
            u_all_pred[k, :] = u_normalized*u_std + u_mean
    
    cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    pic_folder = "pics\\nn_test_"+cur_time
    if not os.path.exists(pic_folder):
         os.mkdir(pic_folder)
    pu.plot_nn_comparison(pic_folder, x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, tra_ref, aero_info, u_pred=u_all_pred)

