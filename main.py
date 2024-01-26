import numpy as np
import dynamics as dyn
import config_opc
import plot_utils as pu
import simulate as simu
import optimal as opt
from matplotlib import pyplot as plt
import time


if __name__ == '__main__':
    cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())

    tra_ref = simu.generate_ref_trajectory_constant(constant_height=300)
    switch_time = 0.5
    tra_ref = simu.generate_ref_trajectory_varying(switch_time=switch_time)
    # x_all_ori, u_all_ori, j_all_ori = simu.simulate_origin(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="pid")
    # plt.plot(x_all_ori[:, 4])
    # plt.show()
    # pu.plot_trajectory_origin(x_all_ori, u_all_ori, j_all_ori, tra_ref)
    x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="pid")
    pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, tra_ref)
    # x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    plt.figure(1)
    plt.plot(x_optimal)
    plt.savefig("pics\\x_optimal_"+cur_time+".png")
    plt.figure(2)
    plt.plot(u_optimal)
    plt.savefig("pics\\u_optimal_"+cur_time+".png")
    plt.figure(3)
    plt.plot(y_optimal)
    plt.savefig("pics\\y_optimal_"+cur_time+".png")
    plt.figure(4)
    plt.plot(z_optimal)
    plt.savefig("pics\\z_optimal_"+cur_time+".png")
    plt.show()
    t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
    pu.plot_trajectory_interpolated(t, x, y, z, u, ref_trajectory=tra_ref)
    
    x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, control_method="given", given_input=u)
    pu.plot_trajectory_auxiliary(x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, tra_ref)

    pass

