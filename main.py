import numpy as np
import dynamics as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from matplotlib import pyplot as plt
import time
import os


if __name__ == '__main__':
    cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())

    # tra_ref = simu.generate_ref_trajectory_constant(constant_height=300)
    switch_time = 0.5
    constant_height=300
    tra_ref = simu.generate_ref_trajectory_varying(constant_height=constant_height, switch_time=switch_time, high_height=350, low_height=250)

#     # x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="pid")
#     # pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, tra_ref, aero_info)
#     # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
#     # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
#     pu.plot_optimal_points(pic_folder, x_optimal, y_optimal, z_optimal, u_optimal)
    t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
#     pu.plot_trajectory_interpolated(pic_folder, t, x, y, z, u, x_optimal, y_optimal, z_optimal, u_optimal, ref_trajectory=tra_ref)
    x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, 
                                                                                                  control_method="given", given_input=u)
    pic_folder = "pics\\results_"+cur_time
    if not os.path.exists(pic_folder):
         os.mkdir(pic_folder)
    pu.plot_trajectory_auxiliary(pic_folder, x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, tra_ref, aero_info)

    # net_path = 'model/net_06-03-2200.pth'
    # x_all_net, y_all_net, z_all_net, u_all_net, j_f_net, aero_info_net = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, 
    #                                                                                              control_method="nn", net_path=net_path, 
    #                                                                                              trajectory_opt=None)
    # # pu.plot_trajectory_auxiliary(pic_folder, x_all_net, y_all_net, z_all_net, u_all_net, j_f_net, tra_ref, aero_info_net)
    # pic_folder = "pics\\compare_"+cur_time
    # if not os.path.exists(pic_folder):
    #      os.mkdir(pic_folder)
    # pu.plot_trajectory_comparison(pic_folder, x_all_net, y_all_net, z_all_net, u_all_net, j_f_net, tra_ref, aero_info_net, u_train=u_all_simu)

    pass