import numpy as np
import dynamics2 as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from data_generate import reload_data, OptimalDataset
from matplotlib import pyplot as plt
import time
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare open loop results.')
    # parser.add_argument('--net_path_both', type=str, default='model/net_02-20-2247/epoch_994.pth')
    # parser.add_argument('--net_path', type=str, default='model/net_02-20-1739/epoch_991.pth')
    # parser.add_argument('--net_path', type=str, default='model/net_02-23-2329/epoch_999.pth')
    parser.add_argument('--net_path', type=str, default='model/net_02-26-1412/epoch_998.pth')
    parser.add_argument('--net_path_both', type=str, default='model/net_02-23-2329/epoch_999.pth')
    parser.add_argument('--file_folder_n', type=str, default="pics/results_open_20250130-1636_0", help='Folder for non-morphing data')
    # parser.add_argument('--file_folder_m', type=str, default="pics/results_open_20250128-1659", help='Folder for morphing data')
    parser.add_argument('--file_folder_m', type=str, default="pics/results_open_20250130-1636_1", help='Folder for morphing data')
    parser.add_argument('--file_folder_n_b', type=str, default="pics/results_open_20250210-0038_7", help='Folder for non-morphing both data')
    parser.add_argument('--file_folder_m_b', type=str, default="pics/results_open_20250210-0038_7", help='Folder for morphing both data')
    parser.add_argument('--file_folder_net', type=str, default="pics/results_close_20250306-1723", help='Folder for morphing both data')
    parser.add_argument('--file_folder_net_b', type=str, default="pics/results_close_20250306-1723", help='Folder for morphing both data')

    args = parser.parse_args()

    net_path = args.net_path
    net_path_both = args.net_path_both
    file_folder_n = args.file_folder_n
    file_folder_m = args.file_folder_m
    file_folder_n_b = args.file_folder_n_b
    file_folder_m_b = args.file_folder_m_b
    file_folder_net = args.file_folder_net
    file_folder_net_b = args.file_folder_net_b

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

    with np.load(f'{file_folder_m}\\data_morphing.npz') as data_morphing:
        keys = ['x_m', 'y_m', 'z_m', 'u_m', 'j_f_m', 'aero_info_m']
        results_m = [data_morphing[key] for key in keys]

    with np.load(f'{file_folder_n_b}\\data_nomorphing_both.npz') as data_nomorphing_both:
        keys = ['x_n_b', 'y_n_b', 'z_n_b', 'u_n_b', 'j_f_n_b', 'aero_info_n_b']
        results_n_b = [data_nomorphing_both[key] for key in keys]

    with np.load(f'{file_folder_m_b}\\data_morphing_both.npz') as data_morphing_both:
        keys = ['x_m_b', 'y_m_b', 'z_m_b', 'u_m_b', 'j_f_m_b', 'aero_info_m_b']
        results_m_b = [data_morphing_both[key] for key in keys]

    # with np.load(f'{file_folder_net}\\data_net.npz') as data_net:
    #     keys = ['x_net', 'y_net', 'z_net', 'u_net', 'j_f_net', 'aero_info_net']
    #     results_net = [data_net[key] for key in keys]

    # with np.load(f'{file_folder_net_b}\\data_net_b.npz') as data_net_b:
    #     keys = ['x_net_b', 'y_net_b', 'z_net_b', 'u_net_b', 'j_f_net_b', 'aero_info_net_b']
    #     results_net_both = [data_net_b[key] for key in keys]

    

    # Generate net control results
    dataset_path = "data/opt_data_02-22-2343.pt"
    dataset_path = "data/opt_data_02-26-1153.pt"
    stat_path = 'data/opt_stats_02-26-1153.npz'
    t_o = reload_data(dataset_path)
    # t_o = results_m_b[0:3]
    results_net = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="nn", net_path=net_path, stat_path=stat_path, trajectory_opt=t_o)
    # Save
    x_net, y_net, z_net, u_net, j_f_net, aero_info_net = results_net
    np.savez(f'{pic_folder}\\data_net.npz', x_net=x_net, y_net=y_net, z_net=z_net, u_net=u_net, j_f_net=j_f_net, aero_info_net=aero_info_net)

    dataset_path_both = "data/opt_data_02-22-2343.pt"
    stat_path_both = 'data/opt_stats_02-22-2343.npz'
    t_o_both = reload_data(dataset_path_both)
    # t_o = results_m_b[0:3]
    results_net_both = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="nn", net_path=net_path_both, stat_path=stat_path_both, trajectory_opt=t_o_both)
    # Save
    x_net_b, y_net_b, z_net_b, u_net_b, j_f_net_b, aero_info_net_b = results_net_both
    np.savez(f'{pic_folder}\\data_net_b.npz', x_net_b=x_net_b, y_net_b=y_net_b, z_net_b=z_net_b, u_net_b=u_net_b, j_f_net_b=j_f_net_b, aero_info_net_b=aero_info_net_b)

    # Plot
    pu.plot_comparison_close(pic_folder=pic_folder, result_nomorphing=results_n, result_morphing=results_m, result_nomorphing_both=results_n_b, result_morphing_both=results_m_b, result_net=results_net, result_net_both=results_net_both, trajectory_ref=tra_ref)
    print(f"Results saved to {pic_folder}")

