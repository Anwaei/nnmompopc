import numpy as np
import torch
from torch import nn
import dynamics2 as dyn
import config_opc
import plot_utils as pu
from modules import OptimalModule
from calculate_utils import cal_mask_mat

def control_origin_constant():
    u_constant = np.array([0.1, 0.8, 0.5])
    return u_constant  # delta_e, delta_T, xi

def control_origin_PID(ep, ei, ed):
    u = config_opc.PARA_KB + config_opc.PARA_KP * ep + config_opc.PARA_KI * ei + config_opc.PARA_KD * ed
    uc = np.clip(u, config_opc.PARA_U_LOWER_BOUND, config_opc.PARA_U_UPPER_BOUND)
    return uc

def control_origin_PID_increment(up, e, ep, epp):
    dt = config_opc.PARA_DT
    u = up + config_opc.PARA_KP*(e-ep) + config_opc.PARA_KI*e*dt + config_opc.PARA_KD*(e-2*ep+epp)/dt
    uc = np.clip(u, config_opc.PARA_U_LOWER_BOUND, config_opc.PARA_U_UPPER_BOUND)
    return uc

def generate_ref_trajectory_constant(constant_height=300):
    h_r_seq = np.ones(shape=config_opc.PARA_STEP_NUM) * constant_height
    t_switch = config_opc.PARA_TF * 0.5
    trajectory_ref = {'h_r_seq': h_r_seq, 't_switch': t_switch}
    return trajectory_ref

def generate_ref_trajectory_varying(constant_height=300, high_height=350, low_height=250, switch_time=0.5, type='triangle'):
    h_r_seq = np.ones(shape=config_opc.PARA_STEP_NUM) * constant_height
    switch_step = int(config_opc.PARA_STEP_NUM * switch_time)
    time_steps = np.arange(config_opc.PARA_STEP_NUM) * config_opc.PARA_DT
    if type == 'triangle':
        one_phase_step = int((config_opc.PARA_STEP_NUM-switch_step)/4)
        h_r_seq[switch_step+0:switch_step+one_phase_step+1] = np.linspace(start=h_r_seq[0], stop=high_height, num=one_phase_step+1)
        h_r_seq[switch_step+one_phase_step:switch_step+one_phase_step*3+1] = np.linspace(start=high_height, stop=low_height, num=2*one_phase_step+1)
        h_r_seq[switch_step+one_phase_step*3:switch_step+one_phase_step*4+1] = np.linspace(start=low_height, stop=constant_height, num=one_phase_step+1)
    elif type == 'sin':
        if high_height - constant_height != constant_height - low_height:
            raise("Sin ref trajectory should have same disatance between high/cons and cons/low.")
        A = high_height - constant_height
        w = 2*np.pi/(switch_step * config_opc.PARA_DT)
        h_r_seq[0:switch_step] = A*np.sin(w*time_steps[0:switch_step]) + constant_height
    else:
        raise("Ref trajectory type error.")
    t_switch = config_opc.PARA_TF * switch_time
    trajectory_ref = {'h_r_seq': h_r_seq, 't_switch': t_switch, 'time_steps':time_steps}
    return trajectory_ref


def simulate_origin(x0, trajectory_ref, control_method):
    nx = config_opc.PARA_NX_ORIGIN
    nu = config_opc.PARA_NU_ORIGIN 
    nj = config_opc.PARA_NJ_ORIGIN
    h_r_seq = trajectory_ref['h_r_seq']
    t_switch = trajectory_ref['t_switch']

    dt = config_opc.PARA_DT 
    step_all = config_opc.PARA_STEP_NUM

    u0 = config_opc.PARA_U0

    x_all = np.zeros((step_all, nx))
    u_all = np.zeros((step_all, nu))
    j_all = np.zeros((step_all, nj))
    f_all = np.zeros((step_all, 4))

    x_all[0, :] = x0
    u_all[0, :] = u0
    j_all[0, :] = dyn.cost_origin(x=x0, u=u0, h_r=h_r_seq[0], t_current=0, t_switch=t_switch)

    # For pid
    err_int = 0
    err_pre = 0

    for k in range(1, step_all):
        x_all[k, :] = dyn.dynamic_origin_one_step(x=x_all[k-1, :], u=u_all[k-1, :])

        if control_method == "pid":
            err_cur = h_r_seq[k] - x_all[k, 4]
            err_int += err_cur*dt
            err_diff = (err_cur - err_pre)/dt
            err_pre = err_cur
            u_all[k, :] = control_origin_PID(err_cur, err_int, err_diff)
            
        else:
            u_all[k, :] = control_origin_constant()

        j_all[k, :] = dyn.cost_origin(x=x_all[k, :], u=u_all[k, :], h_r=h_r_seq[k], t_current=k*dt, t_switch=t_switch)
    
    return x_all, u_all, j_all

def simulate_auxiliary(x0, trajectory_ref, control_method, given_input=None, net_path=None, trajectory_opt=None):
    print("Simulation start. Control method: " + control_method)    
    nx = config_opc.PARA_NX_AUXILIARY
    nu = config_opc.PARA_NU_AUXILIARY
    nj = config_opc.PARA_NJ_AUXILIARY
    ny = config_opc.PARA_NY_AUXILIARY
    nz = config_opc.PARA_NZ_AUXILIARY
    h_r_seq = trajectory_ref['h_r_seq']
    t_switch = trajectory_ref['t_switch']
    time_steps = trajectory_ref['time_steps']

    dt = config_opc.PARA_DT 
    step_all = config_opc.PARA_STEP_NUM

    u0 = config_opc.PARA_U0
    y0 = np.zeros(ny)
    z0 = np.zeros(nz)

    x_all = np.zeros((step_all, nx))
    y_all = np.zeros((step_all, ny))
    z_all = np.zeros((step_all, nz))
    u_all = np.zeros((step_all, nu))
    j_all = np.zeros((step_all, nj))

    x_all[0, :] = x0
    y_all[0, :] = y0
    z_all[0, :] = z0
    u_all[0, :] = u0
    j_f = 0

    aero_forces_all = np.zeros((step_all, 3))  # L, D, M
    L, D, M, T = dyn.aerodynamic_forces(x0, u0)
    aero_forces_all[0, 0] = L
    aero_forces_all[0, 1] = D
    aero_forces_all[0, 2] = M
    aero_deriv_all = np.zeros((step_all, 3))  # CL, CD, CM
    aero_deriv_all[0, :] = dyn.aerodynamic_derivatives(x0, u0)
    angle_deg_all = np.zeros((step_all, 3))  # alpha, q, theta
    angle_deg_all[0, :] = x0[1:4] / np.pi * 180

    # For net
    torch.set_grad_enabled(False)
    net = None
    if control_method == "nn":
        net = OptimalModule()
        net.load_state_dict(torch.load(net_path))
        net.eval()
        opt_stats = np.load(config_opc.STAT_PATH)
        x_mean = opt_stats['x_mean']
        x_std = opt_stats['x_std']
        y_mean = opt_stats['y_mean']
        y_std = opt_stats['y_std']
        z_mean = opt_stats['z_mean']
        z_std = opt_stats['z_std']
        u_mean = opt_stats['u_mean']
        u_std = opt_stats['u_std']
        h_r_mean = opt_stats['h_r_mean']
        h_r_std = opt_stats['h_r_std']
        net_input_ref = (h_r_seq-h_r_mean)/h_r_std

    # For pid
    err_int = 0
    err_cur = h_r_seq[0] - x_all[0, 4]
    err_pre = 0
    err_pre_pre = 0

    # Set u0
    if control_method == "given":
        u_all[0, :] = given_input[0, :]
    elif control_method == "nn":
        x_normalized = (x_all[0, :]-x_mean)/x_std
        y_normalized = (y_all[0, :]-y_mean)/y_std
        z_normalized = (z_all[0, :]-z_mean)/z_std
        h_r_normalized = net_input_ref[0]
        net_input_state = np.concatenate((x_normalized, y_normalized, z_normalized, h_r_normalized[np.newaxis]), axis=0)
        net_input_time = time_steps[0][np.newaxis]/config_opc.PARA_TF
        net_input = np.concatenate((net_input_state, net_input_time, net_input_ref)).astype(np.float32)
        net_input = torch.from_numpy(np.expand_dims(net_input, axis=0))
        mask_mat = cal_mask_mat(net_input)
        u_predict = net(net_input, mask_mat)
        u_normalized = u_predict.detach().numpy().astype(np.float64)
        uc = np.clip(u_normalized*u_std + u_mean, config_opc.PARA_U_LOWER_BOUND, config_opc.PARA_U_UPPER_BOUND)            
        u_all[0, :] = uc

    for k in range(1, step_all):
        x_all[k, :], y_all[k, :], z_all[k, :] = dyn.dynamic_auxiliary_one_step(x=x_all[k-1, :], y=y_all[k-1, :], z=z_all[k-1, :], 
        x_r=h_r_seq[k-1], u=u_all[k-1, :], t=k*dt, t_switch=t_switch)

        if k % 1000 == 0:
            print(f"Simulation step: {k}/{step_all}")

        if control_method == "nn":
            if k == 250:
                pass
            if trajectory_opt is not None:
                x_normalized = (trajectory_opt[0][k, :]-x_mean)/x_std
                y_normalized = (trajectory_opt[1][k, :]-y_mean)/y_std
                z_normalized = (trajectory_opt[2][k, :]-z_mean)/z_std
            else:
                x_normalized = (x_all[k, :]-x_mean)/x_std
                y_normalized = (y_all[k, :]-y_mean)/y_std
                z_normalized = (z_all[k, :]-z_mean)/z_std
            h_r_normalized = net_input_ref[k]

            net_input_state = np.concatenate((x_normalized, y_normalized, z_normalized, h_r_normalized[np.newaxis]), axis=0)
            net_input_time = time_steps[k][np.newaxis]/config_opc.PARA_TF
            net_input = np.concatenate((net_input_state, net_input_time, net_input_ref)).astype(np.float32)
            net_input = torch.from_numpy(np.expand_dims(net_input, axis=0))
            mask_mat = cal_mask_mat(net_input)
            u_predict = net(net_input, mask_mat)
            u_normalized = u_predict.detach().numpy().astype(np.float64)
            uc = np.clip(u_normalized*u_std + u_mean, config_opc.PARA_U_LOWER_BOUND, config_opc.PARA_U_UPPER_BOUND)            
            u_all[k, :] = uc            
            # u_all[k, :] = u_normalized*u_std + u_mean
            pass

        elif control_method == "pid":
            err_pre_pre = err_pre
            err_pre = err_cur
            err_cur = h_r_seq[k] - x_all[k, 4]
            # err_int += err_cur*dt
            # err_diff = (err_cur - err_pre)/dt
            # u_all[k, :] = control_origin_PID(err_cur, err_int, err_diff)
            u_all[k, :] = control_origin_PID_increment(u_all[k-1, :], err_cur, err_pre, err_pre_pre)
        elif control_method == "given":
            if given_input is None: 
                raise("Given input missed.")
            u_all[k, :] = given_input[k, :]
        else:
            u_all[k, :] = control_origin_constant()

        L, D, M, T = dyn.aerodynamic_forces(x_all[k, :], u_all[k, :])
        aero_forces_all[k, 0] = L
        aero_forces_all[k, 1] = D
        aero_forces_all[k, 2] = M
        angle_deg_all[k, :] = x_all[k, 1:4] / np.pi * 180
        aero_deriv_all[k, :] = dyn.aerodynamic_derivatives(x_all[k, :], u_all[k, :])

    
    # j_f = dyn.cost_auxiliary(y_f=y_all[-1, :], z_f=z_all[-1, :], t_switch=t_switch)
    j_f = dyn.cost_auxiliary_all(y_f=y_all[-1, :], z_f=z_all[-1, :], t_switch=t_switch)

    torch.set_grad_enabled(True)
    return x_all, y_all, z_all, u_all, j_f, (aero_forces_all, aero_deriv_all, angle_deg_all)