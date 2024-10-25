import numpy as np
from matplotlib import pyplot as plt
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


def plot_comparison_open_morphing(pic_folder, result_nomorphing, result_morphing, trajectory_ref, from_file = False):
    if from_file:
        pass
    else:
        x_n, y_n, z_n, u_n, j_f_n, aero_info_n = result_nomorphing
        x_m, y_m, z_m, u_m, j_f_m, aero_info_m = result_morphing
        np.savez(f'{pic_folder}\\data_nomorphing.npz', x_n=x_n, y_n=y_n, z_n=z_n, u_n=u_n, j_f_n=j_f_n,
                 aero_info_n=aero_info_n)
        np.savez(f'{pic_folder}\\data_morphing.npz', x_m=x_m, y_m=y_m, z_m=z_m, u_m=u_m, j_f_m=j_f_m,
                 aero_info_m=aero_info_m)
        np.savez(f'{pic_folder}\\h_ref.npz', h_r = trajectory_ref['h_r_seq'])

    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    h_r = trajectory_ref['h_r_seq']
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

    # Main Figure
    plt.figure()
    plt.title("Trajectory Comparison")
    plt.plot(time_steps, np.column_stack((h_r, h_n, h_m)))
    plt.ylim([200, 500])
    plt.xlabel('t')
    plt.ylabel('h')
    plt.legend(['Reference Trajectory', 'No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_tra.png")

    plt.figure()
    plt.title("Tracking Error Comparison")
    err_n = np.sqrt((h_n-h_r)**2)
    err_m = np.sqrt((h_m-h_r)**2)
    plt.plot(time_steps, np.column_stack((err_n, err_m)))
    plt.xlabel('t')
    plt.ylabel('RMSE of h')
    plt.legend(['No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_err.png")

    plt.figure()
    plt.title("Normalized Morphing Parameter Comparison")
    plt.plot(time_steps, np.column_stack((xi_n, xi_m)))
    plt.xlabel('t')
    plt.ylabel(r'$\xi$')
    plt.legend(['No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_xi.png")

    plt.figure()
    plt.title("Normalized Subordinate Objectives")
    plt.subplot(1,2,1)
    plt.plot(time_steps, np.column_stack((y1_n, y1_m)))
    plt.xlabel('t')
    plt.ylabel('y_cruise')
    plt.legend(['No Morphing', 'Morphing'])
    plt.subplot(1,2,2)
    plt.plot(time_steps, np.column_stack((y2_n, y2_m)))
    plt.xlabel('t')
    plt.ylabel('y_aggressive')
    plt.legend(['No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_y.png")

    print("No Morphing: ")
    print(f"Final tracking error: {j_f_n[0] * config_opc.PARA_ERROR_SCALE}")
    print(f"Final cruise cost: {j_f_n[1]}")
    print(f"Final aggressive cost: {j_f_n[2]}")
    print("Morphing: ")
    print(f"Final tracking error: {j_f_m[0] * config_opc.PARA_ERROR_SCALE}")
    print(f"Final cruise cost: {j_f_m[1]}")
    print(f"Final aggressive cost: {j_f_m[2]}")

    # Auxiliary Figures
    plt.figure()
    plt.title("Other Trajectories")
    plt.subplot(2,3,1)
    plt.plot(time_steps, np.column_stack((V_n, V_m)))
    plt.ylabel('V')
    plt.subplot(2,3,2)
    plt.plot(time_steps, np.column_stack((alpha_n, alpha_m)))
    plt.ylabel('alpha')
    plt.subplot(2,3,3)
    plt.plot(time_steps, np.column_stack((q_n, q_m)))
    plt.ylabel('q')
    plt.subplot(2,3,4)
    plt.plot(time_steps, np.column_stack((theta_n, theta_m)))
    plt.ylabel('theta')
    plt.subplot(2,3,5)
    plt.plot(time_steps, np.column_stack((de_n, de_m)))
    plt.ylabel('de')
    plt.subplot(2,3,6)
    plt.plot(time_steps, np.column_stack((T_n, T_m)))
    plt.ylabel('T')
    plt.legend(['No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_others.png")

    aero_forces_n, aero_deriv_n, angle_deg_n = aero_info_n
    aero_forces_m, aero_deriv_m, angle_deg_m = aero_info_m
    plt.figure()
    plt.title("Aerodynamic Coefficients")
    plt.subplot(3, 3, 1)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 0], aero_forces_m[:, 0])))
    plt.ylabel('L')
    plt.subplot(3, 3, 2)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 1], aero_forces_m[:, 1])))
    plt.ylabel('D')
    plt.subplot(3, 3, 3)
    plt.plot(time_steps, np.column_stack((aero_forces_n[:, 2], aero_forces_m[:, 2])))
    plt.ylabel('M')
    plt.subplot(3, 3, 4)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 0], aero_deriv_m[:, 0])))
    plt.ylabel('CL')
    plt.subplot(3, 3, 5)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 1], aero_deriv_m[:, 1])))
    plt.ylabel('CD')
    plt.subplot(3, 3, 6)
    plt.plot(time_steps, np.column_stack((aero_deriv_n[:, 2], aero_deriv_m[:, 2])))
    plt.ylabel('CM')
    plt.subplot(3, 3, 7)
    plt.plot(time_steps, np.column_stack((angle_deg_n[:, 0], angle_deg_m[:, 0])))
    plt.ylabel('alpha')
    plt.subplot(3, 3, 8)
    plt.plot(time_steps, np.column_stack((angle_deg_n[:, 1], angle_deg_m[:, 1])))
    plt.ylabel('q')
    plt.subplot(3, 3, 9)
    plt.plot(time_steps, np.column_stack((angle_deg_n[:, 2], angle_deg_m[:, 2])))
    plt.ylabel('theta')
    plt.legend(['No Morphing', 'Morphing'])
    plt.savefig(pic_folder + "\\cmp_open_morphing_aeroinfo.png")

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
    plt.subplot(2,2,1)
    plt.plot(alpha, CL_0)
    plt.plot(alpha, CL_1)
    plt.plot(alpha, CL_2)
    plt.xlabel("alpha")
    plt.ylabel("CL")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,2)
    plt.plot(alpha, CD_0)
    plt.plot(alpha, CD_1)
    plt.plot(alpha, CD_2)
    plt.xlabel("alpha")
    plt.ylabel("CD")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,3)
    plt.plot(alpha, CM_0)
    plt.plot(alpha, CM_1)
    plt.plot(alpha, CM_2)
    plt.xlabel("alpha")
    plt.ylabel("CM")
    plt.legend([r"$\xi=0$", r"$\xi=0.5$", r"$\xi=1$"])
    plt.subplot(2,2,4)
    plt.plot(alpha, CL_0/CD_0)
    plt.plot(alpha, CL_1/CD_1)
    plt.plot(alpha, CL_2/CD_2)
    plt.xlabel("alpha")
    plt.ylabel("CL/CD")
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
    # test_aerodynamic_coefficient()
    test_collect_points(6)
