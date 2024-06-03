import numpy as np
from matplotlib import pyplot as plt
import time
import os
import config_opc
import optimal as opt

def plot_trajectory_origin(x_all, u_all, j_all, ref_trajectory):
    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    plt.figure(1)
    plt.plot(time_steps, ref_trajectory['h_r_seq'])
    plt.plot(time_steps, x_all)
    plt.legend(['h_ref', 'V', 'gamma', 'q', 'alpha', 'h'])
    plt.figure(2)
    plt.plot(time_steps, u_all)
    plt.legend(['delta_e', 'delta_T', 'xi'])
    plt.figure(3)
    plt.plot(time_steps, j_all[:, 0])
    plt.legend(['J1'])
    plt.figure(4)
    plt.plot(time_steps, j_all[:, 1])
    plt.legend(['J2'])

    plt.show()


def plot_trajectory_auxiliary(pic_folder, x_all, y_all, z_all, u_all, j_f, ref_trajectory, aero_info):
    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    plt.figure()
    plt.title("[Simulate] Trajectories")
    plt.subplot(2,2,1)
    plt.plot(time_steps, ref_trajectory['h_r_seq'])
    plt.plot(time_steps, x_all)
    plt.ylim([-100, 500])
    # plt.legend(['h_ref', 'V', 'gamma', 'q', 'alpha', 'h'])
    plt.legend(['h_ref', 'V', 'alpha', 'q', 'theta', 'h'])
    plt.subplot(2,2,2)
    plt.plot(time_steps, u_all)
    plt.legend(['delta_e', 'delta_T', 'xi'])
    plt.subplot(2,2,3)
    plt.plot(time_steps, y_all)
    plt.legend(['y12', 'y22'])
    plt.subplot(2,2,4)
    plt.plot(time_steps, z_all)
    plt.legend('z')
    # print('J final =' + str(j_f))
    plt.savefig(pic_folder + "\\trajectory.png")

    print(f"Final tracking error: {j_f[0]}")
    print(f"Final cruise cost: {j_f[1]}")
    print(f"Final aggressive cost: {j_f[2]}")

    aero_forces_all, aero_deriv_all, angle_deg_all = aero_info
    plt.figure()
    plt.title("[Simulate] AeroInfo")
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, aero_forces_all)
    plt.legend(['L', 'D', 'M'])
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, aero_deriv_all)
    plt.legend(['CL', 'CD', 'CM'])
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, angle_deg_all)
    plt.legend(['alpha', 'q', 'theta'])

    plt.savefig(pic_folder + "\\aeroinfo.png")

    plt.show()


def plot_trajectory_interpolated(pic_folder, t, x, y, z, u, x_point, y_point, z_point, u_point, ref_trajectory):
    t_switch = ref_trajectory['t_switch']
    LGL_points = opt.calculate_LGL_points(config_opc.PARA_N_LGL_AGGRE)
    LGL_indexex, LGL_time = opt.calculate_LGL_indexes(LGL_points, t_switch)
    scatter_size = 15
    plt.figure()
    plt.title("[Interpolated] Trajectories")
    plt.subplot(3,4,1)  # height
    plt.plot(t, ref_trajectory['h_r_seq'])
    plt.plot(t, x[:, 4])
    plt.scatter(LGL_time, x_point[:, 4], s=scatter_size)
    plt.legend(['Reference Height', 'Actual Height', 'LGL_h'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\h_interpolated.png")
    plt.subplot(3,4,2) # velocity
    plt.plot(t, x[:, 0])
    plt.scatter(LGL_time, x_point[:, 0], s=scatter_size)
    plt.legend(['Actual Velocity', 'LGL_v'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\v_interpolated.png")
    plt.subplot(3,4,3) # gamma
    plt.plot(t, x[:, 1])
    plt.scatter(LGL_time, x_point[:, 1], s=scatter_size)
    plt.legend(['gamma', 'LGL_gamma'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\gamma_interpolated.png")
    plt.subplot(3,4,4) # q
    plt.plot(t, x[:, 2])
    plt.scatter(LGL_time, x_point[:, 2], s=scatter_size)
    plt.legend(['q', 'LGL_q'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\q_interpolated.png")
    plt.subplot(3,4,5) # alpha
    plt.plot(t, x[:, 3])
    plt.scatter(LGL_time, x_point[:, 3], s=scatter_size)
    plt.legend(['alpha', 'LGL_alpha'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\alpha_interpolated.png")
    plt.subplot(3,4,6) # y
    plt.plot(t, y)
    plt.scatter(LGL_time, y_point[:, 0], s=scatter_size)
    plt.scatter(LGL_time, y_point[:, 1], s=scatter_size)
    plt.legend(['y1', 'y2', 'LGL_y1', 'LGL_y2'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\y_interpolated.png")
    plt.subplot(3,4,7) # z
    plt.plot(t, z)
    plt.scatter(LGL_time, z_point, s=scatter_size)
    plt.legend(['z', 'LGL_z'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\z_interpolated.png")
    plt.subplot(3,4,8) # u-delta-e
    plt.plot(t, u[:, 0])
    plt.scatter(LGL_time, u_point[:, 0], s=scatter_size)
    plt.legend(['delta e', 'LGL_deltae'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\de_interpolated.png")
    plt.subplot(3,4,9) # u-delta-T
    plt.plot(t, u[:, 1])
    plt.scatter(LGL_time, u_point[:, 1], s=scatter_size)
    plt.legend(['T', 'LGL_T'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\dT_interpolated.png")
    plt.subplot(3,4,10) # u-xi
    plt.plot(t, u[:, 2])
    plt.scatter(LGL_time, u_point[:, 2], s=scatter_size)
    plt.legend(['xi', 'LGL_xi'])
    # plt.savefig("pics\\interpolated_"+cur_time+"\\xi_interpolated.png")
    plt.savefig(pic_folder + "\\interpolated.png")

def plot_optimal_points(pic_folder, x_optimal, y_optimal, z_optimal, u_optimal):
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x_optimal)
    plt.subplot(2,2,2)
    plt.plot(u_optimal)
    plt.subplot(2,2,3)
    plt.plot(y_optimal)
    plt.subplot(2,2,4)
    plt.plot(z_optimal)
    plt.savefig(pic_folder + "\\optimal_points.png")
    plt.show()

def plot_nn_comparison(pic_folder, x_all, y_all, z_all, u_all, j_f, ref_trajectory, aero_info, u_pred):
    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    plt.figure()
    plt.title("[Simulate] Trajectories")
    plt.subplot(2,2,1)
    plt.plot(time_steps, ref_trajectory['h_r_seq'])
    plt.plot(time_steps, x_all)
    plt.ylim([-100, 500])
    # plt.legend(['h_ref', 'V', 'gamma', 'q', 'alpha', 'h'])
    plt.legend(['h_ref', 'V', 'alpha', 'q', 'theta', 'h'])
    plt.subplot(2,2,2)
    plt.plot(time_steps, u_all)
    plt.plot(time_steps, u_pred)
    plt.legend(['delta_e', 'delta_T', 'xi', '[nn]delta_e', '[nn]delta_T', '[nn]xi'])
    plt.subplot(2,2,3)
    plt.plot(time_steps, y_all)
    plt.legend(['y12', 'y22'])
    plt.subplot(2,2,4)
    plt.plot(time_steps, z_all)
    plt.legend('z')
    # print('J final =' + str(j_f))
    plt.savefig(pic_folder + "\\trajectory.png")

    print(f"Final tracking error: {j_f[0]}")
    print(f"Final cruise cost: {j_f[1]}")
    print(f"Final aggressive cost: {j_f[2]}")

    aero_forces_all, aero_deriv_all, angle_deg_all = aero_info
    plt.figure()
    plt.title("[Simulate] AeroInfo")
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, aero_forces_all)
    plt.legend(['L', 'D', 'M'])
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, aero_deriv_all)
    plt.legend(['CL', 'CD', 'CM'])
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, angle_deg_all)
    plt.legend(['alpha', 'q', 'theta'])

    plt.savefig(pic_folder + "\\aeroinfo.png")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, u_all[:, 0])
    plt.plot(time_steps, u_pred[:, 0])
    plt.legend(['delta_e', '[nn]delta_e'])
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, u_all[:, 1])
    plt.plot(time_steps, u_pred[:, 1])
    plt.legend(['delta_T', '[nn]delta_T'])
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, u_all[:, 2])
    plt.plot(time_steps, u_pred[:, 2])
    plt.legend(['xi', '[nn]xi'])
    plt.savefig(pic_folder + "\\control_comparison.png")


    plt.show()
