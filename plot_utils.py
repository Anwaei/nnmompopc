import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import os
import config_opc
import optimal as opt
import dynamics2 as dyn

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
    plt.legend(['h_ref', 'V', 'gamma', 'q', 'alpha', 'h'])
    # plt.legend(['h_ref', 'V', 'alpha', 'q', 'theta', 'h', 'xi_a'])
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
    # plt.subplot(3,4,11) # x-xi_a
    # plt.plot(t, x[:, 5])
    # plt.scatter(LGL_time, x_point[:, 5], s=scatter_size)
    # plt.legend(['xi_a', 'LGL_xi_a'])
    
    plt.savefig(pic_folder + "\\interpolated.png")

def plot_optimal_points(x_optimal, y_optimal, z_optimal, u_optimal):
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(x_optimal)
    plt.subplot(2,2,2)
    plt.plot(u_optimal)
    plt.subplot(2,2,3)
    plt.plot(y_optimal)
    plt.subplot(2,2,4)
    plt.plot(z_optimal)
    # plt.savefig(pic_folder + "\\optimal_points.png")
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


def plot_trajectory_comparison(pic_folder, x_all, y_all, z_all, u_all, j_f, ref_trajectory, aero_info, u_train):
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
    plt.plot(time_steps, u_train)
    plt.plot(time_steps, u_all)
    plt.ylim([-10, 60])
    plt.legend(['[ps]delta_e', '[ps]delta_T', '[ps]xi', '[nn]delta_e', '[nn]delta_T', '[nn]xi'])
    plt.subplot(2,2,3)
    plt.plot(time_steps, y_all)
    plt.ylim([-100, 100])
    plt.legend(['y12', 'y22'])
    plt.subplot(2,2,4)
    plt.plot(time_steps, z_all)
    plt.ylim([-2, 2])
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
    plt.plot(time_steps, u_train[:, 0])
    plt.plot(time_steps, u_all[:, 0])
    plt.legend(['delta_e', '[nn]delta_e'])
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, u_train[:, 1])
    plt.plot(time_steps, u_all[:, 1])
    plt.legend(['delta_T', '[nn]delta_T'])
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, u_train[:, 2])
    plt.plot(time_steps, u_all[:, 2])
    plt.legend(['[ps]xi', '[nn]xi'])
    plt.savefig(pic_folder + "\\control_comparison.png")

    plt.show()


def plot_comparison_open_morphing(pic_folder=None, 
                                  result_nomorphing=None, result_morphing=None, 
                                  result_nomorphing_fuel=None, result_morphing_fuel=None, 
                                  result_nomorphing_manu=None, result_morphing_manu=None, 
                                  result_nomorphing_both=None, result_morphing_both=None,
                                  trajectory_ref=None, from_file = False, file_folder=None,
                                  shown=True):
    # if from_file:
    #     with np.load(f'{file_folder}\\data_nomorphing.npz') as data_nomorphing:
    #         keys = ['x_n', 'y_n', 'z_n', 'u_n', 'j_f_n', 'aero_info_n']
    #         x_n, y_n, z_n, u_n, j_f_n, aero_info_n = [data_nomorphing[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_morphing.npz') as data_morphing:
    #         keys = ['x_m', 'y_m', 'z_m', 'u_m', 'j_f_m', 'aero_info_m']
    #         x_m, y_m, z_m, u_m, j_f_m, aero_info_m = [data_morphing[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_nomorphing_fuel.npz') as data_nomorphing_fuel:
    #         keys = ['x_n_f', 'y_n_f', 'z_n_f', 'u_n_f', 'j_f_n_f', 'aero_info_n_f']
    #         x_n_f, y_n_f, z_n_f, u_n_f, j_f_n_f, aero_info_n_f = [data_nomorphing_fuel[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_morphing_fuel.npz') as data_morphing_fuel:
    #         keys = ['x_m_f', 'y_m_f', 'z_m_f', 'u_m_f', 'j_f_m_f', 'aero_info_m_f']
    #         x_m_f, y_m_f, z_m_f, u_m_f, j_f_m_f, aero_info_m_f = [data_morphing_fuel[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_nomorphing_manu.npz') as data_nomorphing_manu:
    #         keys = ['x_n_m', 'y_n_m', 'z_n_m', 'u_n_m', 'j_f_n_m', 'aero_info_n_m']
    #         x_n_m, y_n_m, z_n_m, u_n_m, j_f_n_m, aero_info_n_m = [data_nomorphing_manu[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_morphing_manu.npz') as data_morphing_manu:
    #         keys = ['x_m_m', 'y_m_m', 'z_m_m', 'u_m_m', 'j_f_m_m', 'aero_info_m_m']
    #         x_m_m, y_m_m, z_m_m, u_m_m, j_f_m_m, aero_info_m_m = [data_morphing_manu[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_nomorphing_both.npz') as data_nomorphing_both:
    #         keys = ['x_n_b', 'y_n_b', 'z_n_b', 'u_n_b', 'j_f_n_b', 'aero_info_n_b']
    #         x_n_b, y_n_b, z_n_b, u_n_b, j_f_n_b, aero_info_n_b = [data_nomorphing_both[key] for key in keys]
    #     with np.load(f'{file_folder}\\data_morphing_both.npz') as data_morphing_both:
    #         keys = ['x_m_b', 'y_m_b', 'z_m_b', 'u_m_b', 'j_f_m_b', 'aero_info_m_b']
    #         x_m_b, y_m_b, z_m_b, u_m_b, j_f_m_b, aero_info_m_b = [data_morphing_both[key] for key in keys]
    #     with np.load(f'{file_folder}\\h_ref.npz') as h_ref:
    #         h_r = h_ref['h_r']
    # else:
    #     x_n, y_n, z_n, u_n, j_f_n, aero_info_n = result_nomorphing
    #     x_m, y_m, z_m, u_m, j_f_m, aero_info_m = result_morphing
    #     np.savez(f'{pic_folder}\\data_nomorphing.npz', x_n=x_n, y_n=y_n, z_n=z_n, u_n=u_n, j_f_n=j_f_n,
    #              aero_info_n=aero_info_n)
    #     np.savez(f'{pic_folder}\\data_morphing.npz', x_m=x_m, y_m=y_m, z_m=z_m, u_m=u_m, j_f_m=j_f_m,
    #              aero_info_m=aero_info_m)
    #     # np.savez(f'{pic_folder}\\h_ref.npz', h_r = trajectory_ref['h_r_seq'])
    #     x_n_f, y_n_f, z_n_f, u_n_f, j_f_n_f, aero_info_n_f = result_nomorphing_fuel
    #     x_m_f, y_m_f, z_m_f, u_m_f, j_f_m_f, aero_info_m_f = result_morphing_fuel
    #     np.savez(f'{pic_folder}\\data_nomorphing_fuel.npz', x_n_f=x_n_f, y_n_f=y_n_f, z_n_f=z_n_f, u_n_f=u_n_f, j_f_n_f=j_f_n_f,
    #              aero_info_n_f=aero_info_n_f)
    #     np.savez(f'{pic_folder}\\data_morphing_fuel.npz', x_m_f=x_m_f, y_m_f=y_m_f, z_m_f=z_m_f, u_m_f=u_m_f, j_f_m_f=j_f_m_f,
    #              aero_info_m_f=aero_info_m_f)
    #     # np.savez(f'{pic_folder}\\h_ref.npz', h_r = trajectory_ref['h_r_seq'])
    #     x_n_m, y_n_m, z_n_m, u_n_m, j_f_n_m, aero_info_n_m = result_nomorphing_manu
    #     x_m_m, y_m_m, z_m_m, u_m_m, j_f_m_m, aero_info_m_m = result_morphing_manu
    #     np.savez(f'{pic_folder}\\data_nomorphing_manu.npz', x_n_m=x_n_m, y_n_m=y_n_m, z_n_m=z_n_m, u_n_m=u_n_m, j_f_n_m=j_f_n_m,
    #              aero_info_n_m=aero_info_n_m)
    #     np.savez(f'{pic_folder}\\data_morphing_manu.npz', x_m_m=x_m_m, y_m_m=y_m_m, z_m_m=z_m_m, u_m_m=u_m_m, j_f_m_m=j_f_m_m,
    #              aero_info_m_m=aero_info_m_m)
    #     # np.savez(f'{pic_folder}\\h_ref.npz', h_r = trajectory_ref['h_r_seq'])
    #     x_n_b, y_n_b, z_n_b, u_n_b, j_f_n_b, aero_info_n_b = result_nomorphing_both
    #     x_m_b, y_m_b, z_m_b, u_m_b, j_f_m_b, aero_info_m_b = result_morphing_both
    #     np.savez(f'{pic_folder}\\data_nomorphing_both.npz', x_n_b=x_n_b, y_n_b=y_n_b, z_n_b=z_n_b, u_n_b=u_n_b, j_f_n_b=j_f_n_b,
    #              aero_info_n_b=aero_info_n_b)
    #     np.savez(f'{pic_folder}\\data_morphing_both.npz', x_m_b=x_m_b, y_m_b=y_m_b, z_m_b=z_m_b, u_m_b=u_m_b, j_f_m_b=j_f_m_b,
    #              aero_info_m_b=aero_info_m_b)
    #     np.savez(f'{pic_folder}\\h_ref.npz', h_r = trajectory_ref['h_r_seq'])
    #     h_r = trajectory_ref['h_r_seq']

    h_r = trajectory_ref['h_r_seq']
    x_n, y_n, z_n, u_n, j_f_n, aero_info_n = result_nomorphing
    x_m, y_m, z_m, u_m, j_f_m, aero_info_m = result_morphing
    x_n_f, y_n_f, z_n_f, u_n_f, j_f_n_f, aero_info_n_f = result_nomorphing_fuel
    x_m_f, y_m_f, z_m_f, u_m_f, j_f_m_f, aero_info_m_f = result_morphing_fuel
    x_n_m, y_n_m, z_n_m, u_n_m, j_f_n_m, aero_info_n_m = result_nomorphing_manu
    x_m_m, y_m_m, z_m_m, u_m_m, j_f_m_m, aero_info_m_m = result_morphing_manu
    x_n_b, y_n_b, z_n_b, u_n_b, j_f_n_b, aero_info_n_b = result_nomorphing_both
    x_m_b, y_m_b, z_m_b, u_m_b, j_f_m_b, aero_info_m_b = result_morphing_both

    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    V_n = x_n[:, 0]
    V_m = x_m[:, 0]
    alpha_n = x_n[:, 1]
    alpha_m = x_m[:, 1]
    q_n = x_n[:, 2]
    q_m = x_m[:, 2]
    theta_n = x_n[:, 3]
    theta_m = x_m[:, 3]
    h_n = x_n[:, 4]
    h_m = x_m[:, 4]
    de_n = u_n[:, 0]
    de_m = u_m[:, 0]
    T_n = u_n[:, 1]
    T_m = u_m[:, 1]
    xi_n = u_n[:, 2]
    xi_m = u_m[:, 2]
    y1_n = y_n[:, 0]
    y1_m = y_m[:, 0]
    y2_n = y_n[:, 1]
    y2_m = y_m[:, 1]
    V_n_f = x_n_f[:, 0]
    V_m_f = x_m_f[:, 0]
    alpha_n_f = x_n_f[:, 1]
    alpha_m_f = x_m_f[:, 1]
    q_n_f = x_n_f[:, 2]
    q_m_f = x_m_f[:, 2]
    theta_n_f = x_n_f[:, 3]
    theta_m_f = x_m_f[:, 3]
    h_n_f = x_n_f[:, 4]
    h_m_f = x_m_f[:, 4]
    de_n_f = u_n_f[:, 0]
    de_m_f = u_m_f[:, 0]
    T_n_f = u_n_f[:, 1]
    T_m_f = u_m_f[:, 1]
    xi_n_f = u_n_f[:, 2]
    xi_m_f = u_m_f[:, 2]
    y1_n_f = y_n_f[:, 0]
    y1_m_f = y_m_f[:, 0]
    y2_n_f = y_n_f[:, 1]
    y2_m_f = y_m_f[:, 1]
    V_n_m = x_n_m[:, 0]
    V_m_m = x_m_m[:, 0]
    alpha_n_m = x_n_m[:, 1]
    alpha_m_m = x_m_m[:, 1]
    q_n_m = x_n_m[:, 2]
    q_m_m = x_m_m[:, 2]
    theta_n_m = x_n_m[:, 3]
    theta_m_m = x_m_m[:, 3]
    h_n_m = x_n_m[:, 4]
    h_m_m = x_m_m[:, 4]
    de_n_m = u_n_m[:, 0]
    de_m_m = u_m_m[:, 0]
    T_n_m = u_n_m[:, 1]
    T_m_m = u_m_m[:, 1]
    xi_n_m = u_n_m[:, 2]
    xi_m_m = u_m_m[:, 2]
    y1_n_m = y_n_m[:, 0]
    y1_m_m = y_m_m[:, 0]
    y2_n_m = y_n_m[:, 1]
    y2_m_m = y_m_m[:, 1]
    V_n_b = x_n_b[:, 0]
    V_m_b = x_m_b[:, 0]
    alpha_n_b = x_n_b[:, 1]
    alpha_m_b = x_m_b[:, 1]
    q_n_b = x_n_b[:, 2]
    q_m_b = x_m_b[:, 2]
    theta_n_b = x_n_b[:, 3]
    theta_m_b = x_m_b[:, 3]
    h_n_b = x_n_b[:, 4]
    h_m_b = x_m_b[:, 4]
    de_n_b = u_n_b[:, 0]
    de_m_b = u_m_b[:, 0]
    T_n_b = u_n_b[:, 1]
    T_m_b = u_m_b[:, 1]
    xi_n_b = u_n_b[:, 2]
    xi_m_b = u_m_b[:, 2]
    y1_n_b = y_n_b[:, 0]
    y1_m_b = y_m_b[:, 0]
    y2_n_b = y_n_b[:, 1]
    y2_m_b = y_m_b[:, 1]

    xi_n[:] = 0.5
    xi_n_f[:] = 0.5
    xi_n_m[:] = 0.5
    xi_n_b[:] = 0.5

    # Plot figures comparing major and fuel

    # Plot trajectory comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Create the main plot for trajectory comparison
    ax1.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax1.plot(time_steps, np.column_stack((h_n, h_n_f, h_m, h_m_f)))
    ax1.set_ylim([200, 400])
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$h$')
    ax1.set_title('Trajectory Comparison')
    ax1.legend(['Reference Trajectory', 'Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # Add an inset to the first subplot
    ax_inset = inset_axes(ax1, width="40%", height="30%", loc='upper left')
    ax_inset.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax_inset.plot(time_steps, np.column_stack((h_n, h_n_f, h_m, h_m_f)))
    ax_inset.set_xlim(0, 50)
    ax_inset.set_ylim(298, 302)
    ax_inset.yaxis.set_label_position("right")
    ax_inset.yaxis.tick_right()
    # Create the second subplot for tracking error comparison
    err_n = np.sqrt((h_n-h_r)**2)
    err_n_f = np.sqrt((h_n_f-h_r)**2)
    err_m = np.sqrt((h_m-h_r)**2)
    err_m_f = np.sqrt((h_m_f-h_r)**2)
    ax2.plot(time_steps, np.column_stack((np.cumsum(err_n), np.cumsum(err_n_f), np.cumsum(err_m), np.cumsum(err_m_f))))
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'RMSE of $h$')
    ax2.set_title('Tracking Error Comparison')
    ax2.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # Tight and save
    plt.tight_layout()
    plt.savefig(pic_folder + "\\compar_fuel_tra_err.png")

    # Plot fuel consumption
    plt.figure()
    plt.title("Normalized Fuel Consumption Comparison")
    half_time = int(time_steps.shape[0]/2)
    plt.plot(time_steps[0:half_time], np.column_stack((y1_n[0:half_time], y1_n_f[0:half_time], y1_m[0:half_time], y1_m_f[0:half_time])))
    plt.xlabel(r'$t$')
    plt.ylabel('Fuel Consumption')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    plt.savefig(pic_folder + "\\cmp_fuel_consumption.png")

    # Plot control inputs
    plt.figure(figsize=(15, 5))
    t_switch = trajectory_ref['t_switch']
    LGL_points = opt.calculate_LGL_points(config_opc.PARA_N_LGL_AGGRE)
    LGL_indexes, LGL_time = opt.calculate_LGL_indexes(LGL_points, t_switch)
    scatter_size = 15
    plt.title("Control Inputs Comparison")
    plt.subplot(1, 3, 1)
    lines = plt.plot(time_steps, np.column_stack((xi_n, xi_n_f, xi_m, xi_m_f)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\xi$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((xi_n[LGL_indexes], xi_n_f[LGL_indexes], xi_m[LGL_indexes], xi_m_f[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 2)
    lines = plt.plot(time_steps, np.column_stack((T_n, T_n_f, T_m, T_m_f)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$T$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((T_n[LGL_indexes], T_n_f[LGL_indexes], T_m[LGL_indexes], T_m_f[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 3)
    lines = plt.plot(time_steps, np.column_stack((de_n, de_n_f, de_m, de_m_f)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\delta_e$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((de_n[LGL_indexes], de_n_f[LGL_indexes], de_m[LGL_indexes], de_m_f[LGL_indexes])), s=scatter_size, color=colors)
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_control_inputs.png")

    # Plot aerodynamic info
    
    aero_forces_n, aero_deriv_n, angle_deg_n = aero_info_n
    aero_forces_n_f, aero_deriv_n_f, angle_deg_n_f = aero_info_n_f
    aero_forces_m, aero_deriv_m, angle_deg_m = aero_info_m
    aero_forces_m_f, aero_deriv_m_f, angle_deg_m_f = aero_info_m_f

    plt.figure()
    plt.title("Aerodynamic Derivatives")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0])))
    plt.ylabel(r'$C_L$')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    plt.ylabel(r'$C_D$')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_n_f[:, 2], aero_deriv_m[:, 2], aero_deriv_m_f[:, 2])))
    plt.ylabel(r'$C_M$')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0]))/np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    plt.ylabel(r'$C_L/C_D$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroderiv.png")

    plt.figure()
    plt.title("Aerodynamic Forces and Moments")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_f[:, 0], aero_forces_m[:, 0], aero_forces_m_f[:, 0])))
    plt.ylabel(r'$L$')
    plt.ylim([50, 200])
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_n_f[:, 1], aero_forces_m[:, 1], aero_forces_m_f[:, 1])))
    plt.ylabel(r'$D$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_n_f[:, 2], aero_forces_m[:, 2], aero_forces_m_f[:, 2])))
    plt.ylabel(r'$M$')
    plt.ylim([-20, 20])
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0] / aero_forces_n[:, 1], aero_forces_n_f[:, 0] / aero_forces_n_f[:, 1], aero_forces_m[:, 0] / aero_forces_m[:, 1], aero_forces_m_f[:, 0] / aero_forces_m_f[:, 1])))
    plt.ylabel(r'$L/D$')
    plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroforces.png")

    print("Tracking error:")
    print(f"n: {np.sum(err_n)}")
    print(f"n_f: {np.sum(err_n_f)}")
    print(f"m: {np.sum(err_m)}")
    print(f"m_f: {np.sum(err_m_f)}")
    print("Fuel consumption:")
    print(f"n: {y1_n[half_time]}")
    print(f"n_f: {y1_n_f[half_time]}")
    print(f"m: {y1_m[half_time]}")
    print(f"m_f: {y1_m_f[half_time]}")
    print("Normaized z:")
    print(f"n: {z_n[-1]}")
    print(f"n_f: {z_n_f[-1]}")
    print(f"m: {z_m[-1]}")
    print(f"m_f: {z_m_f[-1]}")
    print("Final objectives:")
    print(f"n: {j_f_n}")
    print(f"n_f: {j_f_n_f}")
    print(f"m: {j_f_m}")
    print(f"m_f: {j_f_m_f}")
    
    
    # Plot figures comparing major and manu

    # Plot trajectory comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Create the main plot for trajectory comparison
    ax1.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax1.plot(time_steps, np.column_stack((h_n, h_n_m, h_m, h_m_m)))
    ax1.set_ylim([200, 400])
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$h$')
    ax1.set_title('Trajectory Comparison')
    ax1.legend(['Reference Trajectory', 'Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    # Add an inset to the first subplot
    ax_inset = inset_axes(ax1, width="40%", height="30%", loc='upper left')
    ax_inset.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax_inset.plot(time_steps, np.column_stack((h_n, h_n_m, h_m, h_m_m)))
    ax_inset.set_xlim(0, 50)
    ax_inset.set_ylim(298, 302)
    ax_inset.yaxis.set_label_position("right")
    ax_inset.yaxis.tick_right()
    # Create the second subplot for tracking error comparison
    err_n = np.sqrt((h_n-h_r)**2)
    err_n_m = np.sqrt((h_n_m-h_r)**2)
    err_m = np.sqrt((h_m-h_r)**2)
    err_m_m = np.sqrt((h_m_m-h_r)**2)
    ax2.plot(time_steps, np.column_stack((np.cumsum(err_n), np.cumsum(err_n_m), np.cumsum(err_m), np.cumsum(err_m_m))))
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'RMSE of $h$')
    ax2.set_title('Tracking Error Comparison')
    ax2.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    # Tight and save
    plt.tight_layout()
    plt.savefig(pic_folder + "\\compare_tra_err_manu.png")

    # Plot y2
    plt.figure()
    plt.title("Normalized Maneuverability and Agility Comparison")
    half_time = int(time_steps.shape[0]/2)
    plt.plot(time_steps[half_time:], np.column_stack((y2_n[half_time:], y2_n_m[half_time:], y2_m[half_time:], y2_m_m[half_time:])))
    plt.xlabel(r'$t$')
    plt.ylabel('Maneuverability and agility index')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    plt.savefig(pic_folder + "\\cmp_y2_manu.png")

    # Plot control inputs
    plt.figure(figsize=(15, 5))
    t_switch = trajectory_ref['t_switch']
    LGL_points = opt.calculate_LGL_points(config_opc.PARA_N_LGL_AGGRE)
    LGL_indexes, LGL_time = opt.calculate_LGL_indexes(LGL_points, t_switch)
    scatter_size = 15
    plt.title("Control Inputs Comparison")
    plt.subplot(1, 3, 1)
    lines = plt.plot(time_steps, np.column_stack((xi_n, xi_n_m, xi_m, xi_m_m)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\xi$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((xi_n[LGL_indexes], xi_n_m[LGL_indexes], xi_m[LGL_indexes], xi_m_m[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 2)
    lines = plt.plot(time_steps, np.column_stack((T_n, T_n_m, T_m, T_m_m)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$T$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((T_n[LGL_indexes], T_n_m[LGL_indexes], T_m[LGL_indexes], T_m_m[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 3)
    lines = plt.plot(time_steps, np.column_stack((de_n, de_n_m, de_m, de_m_m)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\delta_e$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((de_n[LGL_indexes], de_n_m[LGL_indexes], de_m[LGL_indexes], de_m_m[LGL_indexes])), s=scatter_size, color=colors)
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_control_inputs_manu.png")

    # Plot aerodynamic info

    aero_forces_n, aero_deriv_n, angle_deg_n = aero_info_n
    aero_forces_n_m, aero_deriv_n_m, angle_deg_n_m = aero_info_n_m
    aero_forces_m, aero_deriv_m, angle_deg_m = aero_info_m
    aero_forces_m_m, aero_deriv_m_m, angle_deg_m_m = aero_info_m_m

    plt.figure()
    plt.title("Aerodynamic Derivatives")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_m[:, 0], aero_deriv_m[:, 0], aero_deriv_m_m[:, 0])))
    plt.ylabel(r'$C_L$')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_m[:, 1], aero_deriv_m[:, 1], aero_deriv_m_m[:, 1])))
    plt.ylabel(r'$C_D$')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_n_m[:, 2], aero_deriv_m[:, 2], aero_deriv_m_m[:, 2])))
    plt.ylabel(r'$C_M$')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_m[:, 0], aero_deriv_m[:, 0], aero_deriv_m_m[:, 0]))/np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_m[:, 1], aero_deriv_m[:, 1], aero_deriv_m_m[:, 1])))
    plt.ylabel(r'$C_L/C_D$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'])
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroderiv_manu.png")

    plt.figure()
    plt.title("Aerodynamic Forces and Moments")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_m[:, 0], aero_forces_m[:, 0], aero_forces_m_m[:, 0])))
    plt.ylabel(r'$L$')
    plt.ylim([50, 200])
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'], loc='upper left')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_n_m[:, 1], aero_forces_m[:, 1], aero_forces_m_m[:, 1])))
    plt.ylabel(r'$D$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'], loc='upper left')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_n_m[:, 2], aero_forces_m[:, 2], aero_forces_m_m[:, 2])))
    plt.ylabel(r'$M$')
    plt.ylim([-20, 20])
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'], loc='upper left')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0] / aero_forces_n[:, 1], aero_forces_n_m[:, 0] / aero_forces_n_m[:, 1], aero_forces_m[:, 0] / aero_forces_m[:, 1], aero_forces_m_m[:, 0] / aero_forces_m_m[:, 1])))
    plt.ylabel(r'$L/D$')
    plt.legend(['Fixed', 'Fixed-M', 'Morphing', 'Morphing-M'], loc='upper left')
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroforces_manu.png")

    print("Tracking error:")
    print(f"n: {np.sum(err_n)}")
    print(f"n_m: {np.sum(err_n_m)}")
    print(f"m: {np.sum(err_m)}")
    print(f"m_m: {np.sum(err_m_m)}")
    print("Fuel consumption:")
    print(f"n: {y1_n[half_time]}")
    print(f"n_m: {y1_n_m[half_time]}")
    print(f"m: {y1_m[half_time]}")
    print(f"m_m: {y1_m_m[half_time]}")
    print("Normaized z:")
    print(f"n: {z_n[-1]}")
    print(f"n_m: {z_n_m[-1]}")
    print(f"m: {z_m[-1]}")
    print(f"m_m: {z_m_m[-1]}")
    print("Final objectives:")
    print(f"n: {j_f_n}")
    print(f"n_m: {j_f_n_m}")
    print(f"m: {j_f_m}")
    print(f"m_m: {j_f_m_m}")


    # Plot figures comparing major and both

    # Plot trajectory comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Create the main plot for trajectory comparison
    ax1.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax1.plot(time_steps, np.column_stack((h_n, h_n_b, h_m, h_m_b)))
    ax1.set_ylim([200, 400])
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$h$')
    ax1.set_title('Trajectory Comparison')
    ax1.legend(['Reference Trajectory', 'Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    # Add an inset to the first subplot
    ax_inset = inset_axes(ax1, width="40%", height="30%", loc='upper left')
    ax_inset.plot(time_steps, h_r, c='k', linestyle='--', linewidth=1.5)
    ax_inset.plot(time_steps, np.column_stack((h_n, h_n_b, h_m, h_m_b)))
    ax_inset.set_xlim(0, 50)
    ax_inset.set_ylim(298, 302)
    ax_inset.yaxis.set_label_position("right")
    ax_inset.yaxis.tick_right()
    # Create the second subplot for tracking error comparison
    err_n = np.sqrt((h_n-h_r)**2)
    err_n_b = np.sqrt((h_n_b-h_r)**2)
    err_m = np.sqrt((h_m-h_r)**2)
    err_m_b = np.sqrt((h_m_b-h_r)**2)
    ax2.plot(time_steps, np.column_stack((np.cumsum(err_n), np.cumsum(err_n_b), np.cumsum(err_m), np.cumsum(err_m_b))))
    ax2.set_xlabel(r'$t$')
    ax2.set_ylabel(r'RMSE of $h$')
    ax2.set_title('Tracking Error Comparison')
    ax2.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    # Tight and save
    plt.tight_layout()
    plt.savefig(pic_folder + "\\compare_tra_err_both.png")

    # Plot fuel consumption
    plt.figure()
    plt.title("Normalized Fuel Consumption Comparison")
    half_time = int(time_steps.shape[0]/2)
    plt.plot(time_steps[0:half_time], np.column_stack((y1_n[0:half_time], y1_n_b[0:half_time], y1_m[0:half_time], y1_m_b[0:half_time])))
    plt.xlabel(r'$t$')
    plt.ylabel('Fuel Consumption')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    plt.savefig(pic_folder + "\\cmp_fuel_consumption_both_.png")

    # Plot y2
    plt.figure()
    plt.title("Normalized Maneuverability and Agility Comparison")
    half_time = int(time_steps.shape[0]/2)
    plt.plot(time_steps[half_time:], np.column_stack((y2_n[half_time:], y2_n_b[half_time:], y2_m[half_time:], y2_m_b[half_time:])))
    plt.xlabel(r'$t$')
    plt.ylabel('Maneuverability and agility index')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    plt.savefig(pic_folder + "\\cmp_y2_both.png")

    # Plot control inputs
    plt.figure(figsize=(15, 5))
    t_switch = trajectory_ref['t_switch']
    LGL_points = opt.calculate_LGL_points(config_opc.PARA_N_LGL_AGGRE)
    LGL_indexes, LGL_time = opt.calculate_LGL_indexes(LGL_points, t_switch)
    scatter_size = 15
    plt.title("Control Inputs Comparison")
    plt.subplot(1, 3, 1)
    lines = plt.plot(time_steps, np.column_stack((xi_n, xi_n_b, xi_m, xi_m_b)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\xi$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((xi_n[LGL_indexes], xi_n_b[LGL_indexes], xi_m[LGL_indexes], xi_m_b[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 2)
    lines = plt.plot(time_steps, np.column_stack((T_n, T_n_b, T_m, T_m_b)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$T$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((T_n[LGL_indexes], T_n_b[LGL_indexes], T_m[LGL_indexes], T_m_b[LGL_indexes])), s=scatter_size, color=colors)
    plt.subplot(1, 3, 3)
    lines = plt.plot(time_steps, np.column_stack((de_n, de_n_b, de_m, de_m_b)))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\delta_e$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    colors = [line.get_color() for line in lines]
    colors = np.tile(colors, (len(LGL_time), 1)).flatten()
    plt.scatter(np.tile(LGL_time, (4, 1)).T, np.column_stack((de_n[LGL_indexes], de_n_b[LGL_indexes], de_m[LGL_indexes], de_m_b[LGL_indexes])), s=scatter_size, color=colors)
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_control_inputs_both.png")

    # Plot aerodynamic info

    aero_forces_n, aero_deriv_n, angle_deg_n = aero_info_n
    aero_forces_n_b, aero_deriv_n_b, angle_deg_n_b = aero_info_n_b
    aero_forces_m, aero_deriv_m, angle_deg_m = aero_info_m
    aero_forces_m_b, aero_deriv_m_b, angle_deg_m_b = aero_info_m_b

    plt.figure()
    plt.title("Aerodynamic Derivatives")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_b[:, 0], aero_deriv_m[:, 0], aero_deriv_m_b[:, 0])))
    plt.ylabel(r'$C_L$')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_b[:, 1], aero_deriv_m[:, 1], aero_deriv_m_b[:, 1])))
    plt.ylabel(r'$C_D$')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_n_b[:, 2], aero_deriv_m[:, 2], aero_deriv_m_b[:, 2])))
    plt.ylabel(r'$C_M$')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_b[:, 0], aero_deriv_m[:, 0], aero_deriv_m_b[:, 0]))/np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_b[:, 1], aero_deriv_m[:, 1], aero_deriv_m_b[:, 1])))
    plt.ylabel(r'$C_L/C_D$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'])
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroderiv_both.png")

    plt.figure()
    plt.title("Aerodynamic Forces and Moments")
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_b[:, 0], aero_forces_m[:, 0], aero_forces_m_b[:, 0])))
    plt.ylabel(r'$L$')
    plt.ylim([50, 200])
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'], loc='upper left')
    plt.subplot(2, 2, 2)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_n_b[:, 1], aero_forces_m[:, 1], aero_forces_m_b[:, 1])))
    plt.ylabel(r'$D$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'], loc='upper left')
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_n_b[:, 2], aero_forces_m[:, 2], aero_forces_m_b[:, 2])))
    plt.ylabel(r'$M$')
    plt.ylim([-20, 20])
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'], loc='upper left')
    plt.subplot(2, 2, 4)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0] / aero_forces_n[:, 1], aero_forces_n_b[:, 0] / aero_forces_n_b[:, 1], aero_forces_m[:, 0] / aero_forces_m[:, 1], aero_forces_m_b[:, 0] / aero_forces_m_b[:, 1])))
    plt.ylabel(r'$L/D$')
    plt.legend(['Fixed', 'Fixed-B', 'Morphing', 'Morphing-B'], loc='upper left')
    plt.tight_layout()
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroforces_both.png")

    print("Tracking error:")
    print(f"n: {np.sum(err_n)}")
    print(f"n_b: {np.sum(err_n_b)}")
    print(f"m: {np.sum(err_m)}")
    print(f"m_b: {np.sum(err_m_b)}")
    print("Fuel consumption:")
    print(f"n: {y1_n[half_time]}")
    print(f"n_b: {y1_n_b[half_time]}")
    print(f"m: {y1_m[half_time]}")
    print(f"m_b: {y1_m_b[half_time]}")
    print("Normaized z:")
    print(f"n: {z_n[-1]}")
    print(f"n_b: {z_n_b[-1]}")
    print(f"m: {z_m[-1]}")
    print(f"m_b: {z_m_b[-1]}")
    print("Final objectives:")
    print(f"n: {j_f_n}")
    print(f"n_b: {j_f_n_b}")
    print(f"m: {j_f_m}")
    print(f"m_b: {j_f_m_b}")



    # Main Figure
    # plt.figure()
    # plt.title("Trajectory Comparison")
    # plt.plot(time_steps, np.column_stack((h_r, h_n, h_m)))
    # plt.ylim([200, 500])
    # plt.xlabel('t')
    # plt.ylabel('h')
    # plt.legend(['Reference Trajectory', 'No Morphing', 'Morphing'])
    # plt.savefig(pic_folder + "\\cmp_open_morphing_tra.png")

    # plt.figure()
    # plt.title("Tracking Error Comparison")
    # err_n = np.sqrt((h_n-h_r)**2)
    # err_m = np.sqrt((h_m-h_r)**2)
    # plt.plot(time_steps, np.column_stack((err_n, err_m)))
    # plt.xlabel('t')
    # plt.ylabel('RMSE of h')
    # plt.legend(['No Morphing', 'Morphing'])
    # plt.savefig(pic_folder + "\\cmp_open_morphing_err.png")

    # plt.figure()
    # plt.title("Normalized Major Objectives")
    # plt.plot(time_steps, np.column_stack((z_n, z_n_f, z_m, z_m_f)))
    # plt.xlabel('t')
    # plt.ylabel('z')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # plt.savefig(pic_folder + "\\cmp_fuel_err_sum.png")

    # plt.figure()
    # plt.title("Normalized Morphing Parameter Comparison")
    # plt.plot(time_steps, np.column_stack((xi_n, xi_n_f, xi_m, xi_m_f)))
    # plt.xlabel('t')
    # plt.ylabel(r'$\xi$')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # plt.savefig(pic_folder + "\\cmp_fuel_xi.png")

    # plt.figure()
    # plt.title("Normalized Subordinate Objectives")
    # plt.subplot(1,2,1)
    # plt.plot(time_steps, np.column_stack((y1_n, y1_m)))
    # plt.xlabel('t')
    # plt.ylabel('y_cruise')
    # plt.legend(['No Morphing', 'Morphing'])
    # plt.subplot(1,2,2)
    # plt.plot(time_steps, np.column_stack((y2_n, y2_m)))
    # plt.xlabel('t')
    # plt.ylabel('y_aggressive')
    # plt.legend(['No Morphing', 'Morphing'])
    # plt.savefig(pic_folder + "\\cmp_open_morphing_y.png")

    # print("No Morphing: ")
    # print(f"Final tracking error: {j_f_n[0] * config_opc.PARA_ERROR_SCALE}")
    # print(f"Final cruise cost: {j_f_n[1]}")
    # print(f"Final aggressive cost: {j_f_n[2]}")
    # print("Morphing: ")
    # print(f"Final tracking error: {j_f_m[0] * config_opc.PARA_ERROR_SCALE}")
    # print(f"Final cruise cost: {j_f_m[1]}")
    # print(f"Final aggressive cost: {j_f_m[2]}")

    # Auxiliary Figures
    # plt.figure()
    # plt.title("Other Trajectories")
    # plt.subplot(2,3,1)
    # plt.plot(time_steps, np.column_stack((V_n, V_n_f, V_m, V_m_f)))
    # plt.ylabel('V')
    # plt.subplot(2,3,2)
    # plt.plot(time_steps, np.column_stack((alpha_n, alpha_n_f, alpha_m, alpha_m_f)))
    # plt.ylabel('alpha')
    # plt.subplot(2,3,3)
    # plt.plot(time_steps, np.column_stack((q_n, q_n_f, q_m, q_m_f)))
    # plt.ylabel('q')
    # plt.subplot(2,3,4)
    # plt.plot(time_steps, np.column_stack((theta_n, theta_n_f, theta_m, theta_m_f)))
    # plt.ylabel('theta')
    # plt.subplot(2,3,5)
    # plt.plot(time_steps, np.column_stack((de_n, de_n_f, de_m, de_m_f)))
    # plt.ylabel('de')
    # plt.subplot(2,3,6)
    # plt.plot(time_steps, np.column_stack((T_n, T_n_f, T_m, T_m_f)))
    # plt.ylabel('T')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # plt.savefig(pic_folder + "\\cmp_open_morphing_others.png")

    # aero_forces_n, aero_deriv_n, angle_deg_n = aero_info_n
    # aero_forces_n_f, aero_deriv_n_f, angle_deg_n_f = aero_info_n_f
    # aero_forces_m, aero_deriv_m, angle_deg_m = aero_info_m
    # aero_forces_m_f, aero_deriv_m_f, angle_deg_m_f = aero_info_m_f
    # plt.figure()
    # plt.title("Aerodynamic Coefficients")
    # plt.subplot(3, 4, 1)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_f[:, 0], aero_forces_m[:, 0], aero_forces_m_f[:, 0])))
    # plt.ylabel('L')
    # plt.subplot(3, 4, 2)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_n_f[:, 1], aero_forces_m[:, 1], aero_forces_m_f[:, 1])))
    # plt.ylabel('D')
    # plt.subplot(3, 4, 3)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_n_f[:, 2], aero_forces_m[:, 2], aero_forces_m_f[:, 2])))
    # plt.ylabel('M')
    # plt.subplot(3, 4, 4)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0])))
    # plt.ylabel('CL')
    # plt.subplot(3, 4, 5)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    # plt.ylabel('CD')
    # plt.subplot(3, 4, 6)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_n_f[:, 2], aero_deriv_m[:, 2], aero_deriv_m_f[:, 2])))
    # plt.ylabel('CM')
    # plt.subplot(3, 4, 7)
    # plt.plot(time_steps, np.column_stack((angle_deg_n[:, 0], angle_deg_n_f[:, 0], angle_deg_m[:, 0], angle_deg_m_f[:, 0])))
    # plt.ylabel('alpha')
    # plt.subplot(3, 4, 8)
    # plt.plot(time_steps, np.column_stack((angle_deg_n[:, 1], angle_deg_n_f[:, 1], angle_deg_m[:, 1], angle_deg_m_f[:, 1])))
    # plt.ylabel('q')
    # plt.subplot(3, 4, 9)
    # plt.plot(time_steps, np.column_stack((angle_deg_n[:, 2], angle_deg_n_f[:, 2], angle_deg_m[:, 2], angle_deg_m_f[:, 2])))
    # plt.ylabel('theta')
    # plt.subplot(3, 4, 10)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_f[:, 0], aero_forces_m[:, 0], aero_forces_m_f[:, 0]))/np.column_stack((aero_forces_n[:, 1], aero_forces_n_f[:, 1], aero_forces_m[:, 1], aero_forces_m_f[:, 1])))
    # plt.ylabel('L/D')
    # plt.subplot(3, 4, 10)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0]))/np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    # plt.ylabel('CL/CD')
    
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # plt.savefig(pic_folder + "\\cmp_open_morphing_aeroinfo.png")

    # plt.figure()
    # plt.title("Aerodynamic Derivatives")
    # plt.subplot(2, 2, 1)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0])))
    # plt.ylabel(r'$C_L$')
    # plt.subplot(2, 2, 2)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    # plt.ylabel(r'$C_D$')
    # plt.subplot(2, 2, 3)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_n_f[:, 2], aero_deriv_m[:, 2], aero_deriv_m_f[:, 2])))
    # plt.ylabel(r'$C_M$')
    # plt.subplot(2, 2, 4)
    # plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_n_f[:, 0], aero_deriv_m[:, 0], aero_deriv_m_f[:, 0]))/np.column_stack((aero_deriv_n[:, 1], aero_deriv_n_f[:, 1], aero_deriv_m[:, 1], aero_deriv_m_f[:, 1])))
    # plt.ylabel(r'$C_L/C_D$')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'])
    # plt.tight_layout()
    # plt.savefig(pic_folder + "\\cmp_open_morphing_aeroderiv.png")

    # plt.figure()
    # plt.title("Aerodynamic Forces and Moments")
    # plt.subplot(2, 2, 1)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_n_f[:, 0], aero_forces_m[:, 0], aero_forces_m_f[:, 0])))
    # plt.ylabel(r'$L$')
    # plt.ylim([50, 200])
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    # plt.subplot(2, 2, 2)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_n_f[:, 1], aero_forces_m[:, 1], aero_forces_m_f[:, 1])))
    # plt.ylabel(r'$D$')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    # plt.subplot(2, 2, 3)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_n_f[:, 2], aero_forces_m[:, 2], aero_forces_m_f[:, 2])))
    # plt.ylabel(r'$M$')
    # plt.ylim([-20, 20])
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    # plt.subplot(2, 2, 4)
    # plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0] / aero_forces_n[:, 1], aero_forces_n_f[:, 0] / aero_forces_n_f[:, 1], aero_forces_m[:, 0] / aero_forces_m[:, 1], aero_forces_m_f[:, 0] / aero_forces_m_f[:, 1])))
    # plt.ylabel(r'$L/D$')
    # plt.legend(['Fixed', 'Fixed-F', 'Morphing', 'Morphing-F'], loc='upper left')
    # plt.tight_layout()
    # plt.savefig(pic_folder + "\\cmp_open_morphing_aeroforces.png")

    if shown:
        plt.show()


def test_aerodynamic_coefficient():
    alpha = np.arange(start=-10, step=0.1, stop=20)  # deg
    alpha_rad = alpha/180*np.pi
    delta_e = 0
    CL_0 = np.zeros(shape=alpha.shape)
    CD_0 = np.zeros(shape=alpha.shape)
    CM_0 = np.zeros(shape=alpha.shape)
    CL_1 = np.zeros(shape=alpha.shape)
    CD_1 = np.zeros(shape=alpha.shape)
    CM_1 = np.zeros(shape=alpha.shape)
    CL_2 = np.zeros(shape=alpha.shape)
    CD_2 = np.zeros(shape=alpha.shape)
    CM_2 = np.zeros(shape=alpha.shape)
    for i in range(alpha.size):
        CL_0[i] = dyn.aerodynamic_coefficient_lift(alpha=alpha_rad[i], xi=0, delta_e=delta_e)
        CD_0[i] = dyn.aerodynamic_coefficient_drag(alpha=alpha_rad[i], xi=0)
        CM_0[i] = dyn.aerodynamic_coefficient_pitch_moment(alpha=alpha_rad[i], xi=0, delta_e=delta_e)
        CL_1[i] = dyn.aerodynamic_coefficient_lift(alpha=alpha_rad[i], xi=0.5, delta_e=delta_e)
        CD_1[i] = dyn.aerodynamic_coefficient_drag(alpha=alpha_rad[i], xi=0.5)
        CM_1[i] = dyn.aerodynamic_coefficient_pitch_moment(alpha=alpha_rad[i], xi=0.5, delta_e=delta_e)
        CL_2[i] = dyn.aerodynamic_coefficient_lift(alpha=alpha_rad[i], xi=1, delta_e=delta_e)
        CD_2[i] = dyn.aerodynamic_coefficient_drag(alpha=alpha_rad[i], xi=1)
        CM_2[i] = dyn.aerodynamic_coefficient_pitch_moment(alpha=alpha_rad[i], xi=1, delta_e=delta_e)
    
    plt.figure()
    plt.subplot(3,3,1)
    plt.plot(alpha, CL_0)
    plt.xlabel("alpha")
    plt.ylabel("CL")
    plt.legend(["xi=0"])
    plt.subplot(3,3,4)
    plt.plot(alpha, CD_0)
    plt.xlabel("alpha")
    plt.ylabel("CD")
    plt.legend(["xi=0"])
    plt.subplot(3,3,7)
    plt.plot(alpha, CM_0)
    plt.xlabel("alpha")
    plt.ylabel("CM")
    plt.legend(["xi=0"])

    plt.subplot(3,3,2)
    plt.plot(alpha, CL_1)
    plt.xlabel("alpha")
    plt.ylabel("CL")
    plt.legend(["xi=0.5"])
    plt.subplot(3,3,5)
    plt.plot(alpha, CD_1)
    plt.xlabel("alpha")
    plt.ylabel("CD")
    plt.legend(["xi=0.5"])
    plt.subplot(3,3,8)
    plt.plot(alpha, CM_1)
    plt.xlabel("alpha")
    plt.ylabel("CM")
    plt.legend(["xi=0.5"])

    plt.subplot(3,3,3)
    plt.plot(alpha, CL_2)
    plt.xlabel("alpha")
    plt.ylabel("CL")
    plt.legend(["xi=1"])
    plt.subplot(3,3,6)
    plt.plot(alpha, CD_2)
    plt.xlabel("alpha")
    plt.ylabel("CD")
    plt.legend(["xi=1"])
    plt.subplot(3,3,9)
    plt.plot(alpha, CM_2)
    plt.xlabel("alpha")
    plt.ylabel("CM")
    plt.legend(["xi=1"])

    plt.figure()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.subplot(2,2,1)
    plt.plot(alpha, CL_0)
    plt.plot(alpha, CL_1)
    plt.plot(alpha, CL_2)
    plt.xlabel(r"$\alpha$/deg")
    plt.ylabel(r"$C_L$")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,2)
    plt.plot(alpha, CD_0)
    plt.plot(alpha, CD_1)
    plt.plot(alpha, CD_2)
    plt.xlabel(r"$\alpha$/deg")
    plt.ylabel(r"$C_D$")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,3)
    plt.plot(alpha, CM_0)
    plt.plot(alpha, CM_1)
    plt.plot(alpha, CM_2)
    plt.xlabel(r"$\alpha$/deg")
    plt.ylabel(r"$C_M$")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,4)
    plt.plot(alpha, CL_0/CD_0)
    plt.plot(alpha, CL_1/CD_1)
    plt.plot(alpha, CL_2/CD_2)
    plt.xlabel(r"$\alpha$/deg")
    plt.ylabel(r"$C_L/C_D$")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    
    plt.show()


def test_collect_points(N=8):
    LGL_points = opt.calculate_LGL_points(N)
    LGR_points = opt.calculate_LGR_points(N)
    print("LGL_points:")
    print(LGL_points)
    print("LGR_points:")
    print(LGR_points)

    plt.figure(1)
    plt.scatter(LGL_points, np.zeros(N))
    plt.scatter(LGR_points, np.ones(N))
    plt.ylim([-4, 5])
    plt.legend(['LGL points', 'LGR points'])
    plt.title("Collection Points")

    dt = 0.001
    tau = np.arange(start=-1, step=dt, stop=1)

    # ===================================================================== #
    # ===================================================================== #
    # Lagrange Polynomians for LGL points
    plt.figure(2)
    plt.title("Lagrange Polynomians for LGL points")
    L = np.ones((N, tau.size))
    L_point = np.ones((N, N))
    for i in range(N):
        for t in range(tau.size):
            for k in range(N):
                if k != i:
                    L[i, t] *= (tau[t]-LGL_points[k])/(LGL_points[i]-LGL_points[k])# L_i(tau) = Pi(k=0-N, k not i) ((x-x_k)/(x_i-x_k))
        for j in range(N):
            for k in range(N):
                if k != i:
                    L_point[i, j] *= (LGL_points[j]-LGL_points[k])/(LGL_points[i]-LGL_points[k])
    for i in range(N):
        plt.plot(tau, L[i, :], label=f'L_{i}')
        plt.scatter(LGL_points, L_point[i, :])
    plt.legend()
    plt.xlabel('tau')

    plt.figure(3)
    plt.title("Lagrange Differential for LGL points")
    
    # Numerical Differential for LGL points
    plt.subplot(2,2,1)
    plt.title("Numerical Differential")
    Ld_num = np.zeros((N, tau.size))
    Ld_num_point = np.zeros((N, N))
    for i in range(N):
        pi = 0
        for t in range(tau.size-1):
            Ld_num[i, t] = (L[i, t+1] - L[i, t])/dt
            if abs(LGL_points[pi] - tau[t]) <= dt/2 or t == tau.size-2:
                Ld_num_point[i, pi] = Ld_num[i, t]
                pi = pi + 1
    for i in range(N):
        plt.plot(tau[0:-1], Ld_num[i, 0:-1], label=f'Ld_{i}')
        plt.scatter(LGL_points, Ld_num_point[i, :])
    plt.legend()

    # Formula 2
    plt.subplot(2,2,2)   
    plt.title("Formula 2")
    Ld_2 = np.zeros((N, tau.size))
    Ld_2_point = np.zeros((N, N))
    # Realization of formula 3
    # for i in range(N):
    #     for t in range(tau.size-1):
    #         sum = 0
    #         for k in range(N):
    #             if k != i:
    #                 sum += 1/(tau[t]-LGL_points[k])
    #         Ld_2[i, t] = L[i, t] * sum
    #     for j in range(N):
    #         sum = 0
    #         for k in range(N):
    #             if k != i:
    #                 sum += 1/(LGL_points[j]-LGL_points[k])
    #         Ld_2_point[i, j] = L_point[i, j] * sum
    for i in range(N):
        for t in range(tau.size):
            num = 0
            den = 1
            for k in range(N):
                temp = 1
                for l in range(N):
                    if l != k:
                        temp *= (tau[t] - LGL_points[l])
                num += temp
            for k in range(N):
                if k != i:
                    den *= (LGL_points[i] - LGL_points[k])
            Ld_2[i, t] = num/den
        for j in range(N):
            num = 0
            den = 1
            for k in range(N):
                temp = 1
                for l in range(N):
                    if l != k:
                        temp *= (LGL_points[j] - LGL_points[l])
                num += temp
            for k in range(N):
                if k != i:
                    den *= (LGL_points[i] - LGL_points[k])
            Ld_2_point[i, j] = num/den
    for i in range(N):
        plt.plot(tau, Ld_2[i, :], label=f'Ld_{i}')
        plt.scatter(LGL_points, Ld_2_point[i, :])
    plt.legend()            
    
    # Formula 4
    plt.subplot(2,2,3)
    plt.title("Formula 4")
    # Ld_4 = Ld_num.copy()
    Ld_4 = np.zeros((N, tau.size))
    Ld_4_point = np.zeros((N, N))
    for i in range(N):
        for t in range(tau.size):
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (tau[t]-LGL_points[l])/(LGL_points[i]-LGL_points[l])                        
                    Ld_4[i, t] += 1/(LGL_points[i]-LGL_points[j]) * temp
        for k in range(N):
            Dki = 0
            if i == k:
                for j in range(N):
                    if j != i:
                        Dki += 1/(LGL_points[i]-LGL_points[j])
            else:
                num = 1
                for j in range(N):
                    if j != i and j != k:
                        num *= LGL_points[k] - LGL_points[j]
                den = 1
                for j in range(N):
                    if j != i :
                        den *= LGL_points[i] - LGL_points[j]
                Dki = num/den
            Ld_4_point[i, k] = Dki        
    for i in range(N):
        plt.plot(tau, Ld_4[i, :], label=f'Ld_{i}')
        plt.scatter(LGL_points, Ld_4_point[i, :])
    plt.legend()   

    # Formula 4 (dyn2)
    plt.subplot(2,2,4)   
    plt.title("Formula 4 (dyn2)")
    # Ld_3 = Ld_num.copy()
    Ld_3 = np.zeros((N, tau.size))
    Ld_3_point = np.zeros((N, N))
    for i in range(N):
        for t in range(tau.size):
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (tau[t]-LGL_points[l])/(LGL_points[i]-LGL_points[l])                        
                    Ld_3[i, t] += 1/(LGL_points[i]-LGL_points[j]) * temp
        for k in range(N):
            Dki = 0
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (LGL_points[k]-LGL_points[l])/(LGL_points[i]-LGL_points[l])                        
                    Dki += 1/(LGL_points[i]-LGL_points[j]) * temp
            Ld_3_point[i, k] = Dki        
    for i in range(N):
        plt.plot(tau, Ld_3[i, :], label=f'Ld_{i}')
        plt.scatter(LGL_points, Ld_3_point[i, :])
    plt.legend()              

    # ===================================================================== #
    # ===================================================================== #
    # Lagrange Polynomians for LGR points
    plt.figure(4)
    plt.title("Lagrange Polynomians for LGR points")
    L = np.ones((N, tau.size))
    L_point = np.ones((N, N))
    for i in range(N):
        for t in range(tau.size):
            for k in range(N):
                if k != i:
                    L[i, t] *= (tau[t]-LGR_points[k])/(LGR_points[i]-LGR_points[k])# L_i(tau) = Pi(k=0-N, k not i) ((x-x_k)/(x_i-x_k))
        for j in range(N):
            for k in range(N):
                if k != i:
                    L_point[i, j] *= (LGR_points[j]-LGR_points[k])/(LGR_points[i]-LGR_points[k])
    for i in range(N):
        plt.plot(tau, L[i, :], label=f'L_{i}')
        plt.scatter(LGR_points, L_point[i, :])
    plt.legend()
    plt.xlabel('tau')

    plt.figure(5)
    plt.title("Lagrange Differential for LGR points")
    
    # Numerical Differential for LGR points
    plt.subplot(2,2,1)
    plt.title("Numerical Differential")
    Ld_num = np.zeros((N, tau.size))
    Ld_num_point = np.zeros((N, N))
    for i in range(N):
        pi = 0
        for t in range(tau.size-1):
            Ld_num[i, t] = (L[i, t+1] - L[i, t])/dt
            if pi<N and abs(LGR_points[pi] - tau[t]) <= dt/2:
                Ld_num_point[i, pi] = Ld_num[i, t]
                pi = pi + 1
    for i in range(N):
        plt.plot(tau[0:-1], Ld_num[i, 0:-1], label=f'Ld_{i}')
        plt.scatter(LGR_points, Ld_num_point[i, :])
    plt.legend()

    # Formula 2
    plt.subplot(2,2,2)   
    plt.title("Formula 2")
    Ld_2 = np.zeros((N, tau.size))
    Ld_2_point = np.zeros((N, N))
    # Realization of formula 3
    # for i in range(N):
    #     for t in range(tau.size-1):
    #         sum = 0
    #         for k in range(N):
    #             if k != i:
    #                 sum += 1/(tau[t]-LGR_points[k])
    #         Ld_2[i, t] = L[i, t] * sum
    #     for j in range(N):
    #         sum = 0
    #         for k in range(N):
    #             if k != i:
    #                 sum += 1/(LGR_points[j]-LGR_points[k])
    #         Ld_2_point[i, j] = L_point[i, j] * sum
    for i in range(N):
        for t in range(tau.size):
            num = 0
            den = 1
            for k in range(N):
                temp = 1
                for l in range(N):
                    if l != k:
                        temp *= (tau[t] - LGR_points[l])
                num += temp
            for k in range(N):
                if k != i:
                    den *= (LGR_points[i] - LGR_points[k])
            Ld_2[i, t] = num/den
        for j in range(N):
            num = 0
            den = 1
            for k in range(N):
                temp = 1
                for l in range(N):
                    if l != k:
                        temp *= (LGR_points[j] - LGR_points[l])
                num += temp
            for k in range(N):
                if k != i:
                    den *= (LGR_points[i] - LGR_points[k])
            Ld_2_point[i, j] = num/den
    for i in range(N):
        plt.plot(tau, Ld_2[i, :], label=f'Ld_{i}')
        plt.scatter(LGR_points, Ld_2_point[i, :])
    plt.legend()
    
    # Formula 4
    plt.subplot(2,2,3)
    plt.title("Formula 4")
    # Ld_4 = Ld_num.copy()
    Ld_4 = np.zeros((N, tau.size))
    Ld_4_point = np.zeros((N, N))
    for i in range(N):
        for t in range(tau.size):
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (tau[t]-LGR_points[l])/(LGR_points[i]-LGR_points[l])                        
                    Ld_4[i, t] += 1/(LGR_points[i]-LGR_points[j]) * temp
        for k in range(N):
            Dki = 0
            if i == k:
                for j in range(N):
                    if j != i:
                        Dki += 1/(LGR_points[i]-LGR_points[j])
            else:
                num = 1
                for j in range(N):
                    if j != i and j != k:
                        num *= LGR_points[k] - LGR_points[j]
                den = 1
                for j in range(N):
                    if j != i :
                        den *= LGR_points[i] - LGR_points[j]
                Dki = num/den
            Ld_4_point[i, k] = Dki        
    for i in range(N):
        plt.plot(tau, Ld_4[i, :], label=f'Ld_{i}')
        plt.scatter(LGR_points, Ld_4_point[i, :])
    plt.legend()   

    # Formula 4 (dyn2)
    plt.subplot(2,2,4)   
    plt.title("Formula 4 (dyn2)")
    # Ld_3 = Ld_num.copy()
    Ld_3 = np.zeros((N, tau.size))
    Ld_3_point = np.zeros((N, N))
    for i in range(N):
        for t in range(tau.size):
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (tau[t]-LGR_points[l])/(LGR_points[i]-LGR_points[l])                        
                    Ld_3[i, t] += 1/(LGR_points[i]-LGR_points[j]) * temp
        for k in range(N):
            Dki = 0
            for j in range(N):
                if j != i:
                    temp = 1
                    for l in range(N):
                        if l!=i and l!=j:
                            temp *= (LGR_points[k]-LGR_points[l])/(LGR_points[i]-LGR_points[l])                        
                    Dki += 1/(LGR_points[i]-LGR_points[j]) * temp
            Ld_3_point[i, k] = Dki        
    for i in range(N):
        plt.plot(tau, Ld_3[i, :], label=f'Ld_{i}')
        plt.scatter(LGR_points, Ld_3_point[i, :])
    plt.legend()

    plt.show()


if __name__ == "__main__":
    test_aerodynamic_coefficient()
    # test_collect_points(6)
