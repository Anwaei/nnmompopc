import numpy as np
from matplotlib import pyplot as plt
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
