import numpy as np
from matplotlib import pyplot as plt
import time
import os
import config_opc

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


def plot_trajectory_auxiliary(x_all, y_all, z_all, u_all, j_f, ref_trajectory):
    time_steps = np.arange(start=0, stop=config_opc.PARA_TF+config_opc.PARA_DT, step=config_opc.PARA_DT)
    plt.figure()
    plt.plot(time_steps, ref_trajectory['h_r_seq'])
    plt.plot(time_steps, x_all)
    plt.ylim([-200, 1000])
    plt.legend(['h_ref', 'V', 'gamma', 'q', 'alpha', 'h'])
    plt.figure()
    plt.plot(time_steps, u_all)
    plt.legend(['delta_e', 'delta_T', 'xi'])
    plt.figure()
    plt.plot(time_steps, y_all)
    plt.legend(['y12', 'y22'])
    plt.figure()
    plt.plot(time_steps, z_all)
    plt.legend('z')
    print('J final =' + str(j_f))

    plt.show()


def plot_trajectory_interpolated(t, x, y, z, u, ref_trajectory):
    cur_time = time.strftime("%Y%m%d-%H%M", time.localtime())
    if not os.path.exists("pics\\interpolated_"+cur_time):
         os.mkdir("pics\\interpolated_"+cur_time)
    plt.figure()  # height
    plt.plot(t, ref_trajectory['h_r_seq'])
    plt.plot(t, x[:, 4])
    plt.legend(['Reference Height', 'Actual Height'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\h_interpolated.png")
    plt.figure() # velocity
    plt.plot(t, x[:, 0])
    plt.legend(['Actual Height'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\v_interpolated.png")
    plt.figure() # gamma
    plt.plot(t, x[:, 1])
    plt.legend(['gamma'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\gamma_interpolated.png")
    plt.figure() # q
    plt.plot(t, x[:, 2])
    plt.legend(['q'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\q_interpolated.png")
    plt.figure() # alpha
    plt.plot(t, x[:, 3])
    plt.legend(['alpha'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\alpha_interpolated.png")
    plt.figure() # y
    plt.plot(t, y)
    plt.legend(['y1', 'y2'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\y_interpolated.png")
    plt.figure() # z
    plt.plot(t, z)
    plt.legend(['z'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\z_interpolated.png")
    plt.figure() # u-delta-e
    plt.plot(t, u[:, 0])
    plt.legend(['delta e'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\de_interpolated.png")
    plt.figure() # u-delta-T
    plt.plot(t, u[:, 1])
    plt.legend(['delta T'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\dT_interpolated.png")
    plt.figure() # u-xi
    plt.plot(t, u[:, 2])
    plt.legend(['xi'])
    plt.savefig("pics\\interpolated_"+cur_time+"\\xi_interpolated.png")

