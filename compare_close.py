import numpy as np
import dynamics2 as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from matplotlib import pyplot as plt
import time
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare open loop results.')
    # parser.add_argument('--net_path', type=str, default='model/net_02-20-2247/epoch_994.pth')
    parser.add_argument('--net_path', type=str, default='model/net_02-20-1739/epoch_991.pth')
    parser.add_argument('--file_folder_n', type=str, default="pics/results_open_20250128-1659", help='Folder for non-morphing data')
    parser.add_argument('--file_folder_m_b', type=str, default="pics/results_open_20250210-0038_7", help='Folder for morphing fuel data')
    args = parser.parse_args()

    net_path = args.net_path
    file_folder_n = args.file_folder_n
    file_folder_m_b = args.file_folder_m_b

    cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    pic_folder = "pics\\results_close_"+cur_time
    if not os.path.exists(pic_folder):
         os.mkdir(pic_folder)
    switch_time = 0.5
    constant_height=300
    tra_ref = simu.generate_ref_trajectory_varying(constant_height=constant_height, switch_time=switch_time,high_height=350, low_height=250)
    np.savez(f'{pic_folder}\\h_ref.npz', h_r = tra_ref['h_r_seq'])

    with np.load(f'{file_folder_n}\\data_nomorphing.npz') as data_nomorphing:
        keys = ['x_n', 'y_n', 'z_n', 'u_n', 'j_f_n', 'aero_info_n']
        results_n = [data_nomorphing[key] for key in keys]

    with np.load(f'{file_folder_m_b}\\data_morphing_both.npz') as data_morphing_both:
        keys = ['x_m_b', 'y_m_b', 'z_m_b', 'u_m_b', 'j_f_m_b', 'aero_info_m_b']
        results_m_b = [data_morphing_both[key] for key in keys]

    with np.load(f'{file_folder_m_b}\\data_nomorphing_both.npz') as data_nomorphing_both:
        keys = ['x_n_b', 'y_n_b', 'z_n_b', 'u_n_b', 'j_f_n_b', 'aero_info_n_b']
        results_n_b = [data_nomorphing_both[key] for key in keys]

    # Generate net control results
    t_o = results_m_b[0:3]
    results_net = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="nn", net_path=net_path, trajectory_opt=t_o)
    # Save
    x_net, y_net, z_net, u_net, j_f_net, aero_info_net = results_net
    np.savez(f'{pic_folder}\\data_net.npz', x_net=x_net, y_net=y_net, z_net=z_net, u_net=u_net, j_f_net=j_f_net, aero_info_net=aero_info_net)

    # Plot
    pu.plot_comparison_close(pic_folder=pic_folder, result_nomorphing=results_n, result_morphing_both=results_m_b, result_net=results_net, trajectory_ref=tra_ref)
    print(f"Results saved to {pic_folder}")

