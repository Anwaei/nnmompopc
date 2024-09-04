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
    x_optimal, y_optimal, z_optimal, u_optimal, j_optimal = opt.generate_PS_solution_casadi(x0=config_opc.PARA_X0, trajectory_ref=tra_ref)
    t, x, y, z, u = opt.interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch=tra_ref["t_switch"])
    x_all_simu, y_all_simu, z_all_simu, u_all_simu, j_f_simu, aero_info = simu.simulate_auxiliary(x0=config_opc.PARA_X0, trajectory_ref=tra_ref, 
                                                                                                  control_method="given", given_input=u)
    pic_folder = "pics\\results_"+cur_time
    if not os.path.exists(pic_folder):
         os.mkdir(pic_folder)
