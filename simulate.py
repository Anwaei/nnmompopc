import numpy as np
import dynamics as dyn
import config_opc
import plot_utils as pu

def control_origin_constant():
    u_constant = np.array([0.1, 0.8, 0.5])
    return u_constant  # delta_e, delta_T, xi

def control_origin_PID(ep, ei, ed):
    u = config_opc.PARA_KB + config_opc.PARA_KP * ep + config_opc.PARA_KI * ei + config_opc.PARA_KD * ed
    uc = np.clip(u, 0, 1.5)
    return uc

def generate_ref_trajectory_constant(constant_height=300):
    h_r_seq = np.ones(shape=config_opc.PARA_STEP_NUM) * constant_height
    t_switch = 0.5
    trajectory_ref = {'h_r_seq': h_r_seq, 't_switch': t_switch}
    return trajectory_ref

def generate_ref_trajectory_varying(constant_height=300, high_height=400, low_height=200, switch_time=0.5, type='triangle'):
    h_r_seq = np.ones(shape=config_opc.PARA_STEP_NUM) * constant_height
    switch_step = int(config_opc.PARA_STEP_NUM * switch_time)
    time_steps = np.arange(config_opc.PARA_STEP_NUM) * config_opc.PARA_DT
    if type == 'triangle':
        one_phase_step = int(switch_step/4)
        h_r_seq[0:one_phase_step+1] = np.linspace(start=h_r_seq[0], stop=high_height, num=one_phase_step+1)
        h_r_seq[one_phase_step:one_phase_step*3+1] = np.linspace(start=high_height, stop=low_height, num=2*one_phase_step+1)
        h_r_seq[one_phase_step*3:one_phase_step*4+1] = np.linspace(start=low_height, stop=constant_height, num=one_phase_step+1)
    elif type == 'sin':
        if high_height - constant_height != constant_height - low_height:
            raise("Sin ref trajectory should have same disatance between high/cons and cons/low.")
        A = high_height - constant_height
        w = 2*np.pi/(switch_step * config_opc.PARA_DT)
        h_r_seq[0:switch_step] = A*np.sin(w*time_steps[0:switch_step]) + constant_height
    else:
        raise("Ref trajectory type error.")
    t_switch = config_opc.PARA_TF * switch_time
    trajectory_ref = {'h_r_seq': h_r_seq, 't_switch': t_switch}
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
            err_int += err_cur
            err_diff = (err_cur - err_pre)/dt
            err_pre = err_cur
            u_all[k, :] = control_origin_PID(err_cur, err_int, err_diff)
            
        else:
            u_all[k, :] = control_origin_constant()

        j_all[k, :] = dyn.cost_origin(x=x_all[k, :], u=u_all[k, :], h_r=h_r_seq[k], t_current=k*dt, t_switch=t_switch)
    
    return x_all, u_all, j_all

def simulate_auxiliary(x0, trajectory_ref, control_method, given_input=None):
    nx = config_opc.PARA_NX_AUXILIARY
    nu = config_opc.PARA_NU_AUXILIARY
    nj = config_opc.PARA_NJ_AUXILIARY
    ny = config_opc.PARA_NY_AUXILIARY
    nz = config_opc.PARA_NZ_AUXILIARY
    h_r_seq = trajectory_ref['h_r_seq']
    t_switch = trajectory_ref['t_switch']

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

    # For pid
    err_int = 0
    err_pre = 0

    for k in range(1, step_all):
        x_all[k, :], y_all[k, :], z_all[k, :] = dyn.dynamic_auxiliary_one_step(x=x_all[k-1, :], y=y_all[k-1, :], z=z_all[k-1, :], 
        x_r=h_r_seq[k-1], u=u_all[k-1, :], t=k*dt, t_switch=t_switch)

        if control_method == "pid":
            err_cur = h_r_seq[k] - x_all[k, 4]
            err_int += err_cur
            err_diff = (err_cur - err_pre)/dt
            err_pre = err_cur
            u_all[k, :] = control_origin_PID(err_cur, err_int, err_diff)            
        elif control_method == "given":
            if given_input is None: 
                raise("Given input missed.")
            u_all[k, :] = given_input[k, :]
        else:
            u_all[k, :] = control_origin_constant()
    
    j_f = dyn.cost_auxiliary(y_f=y_all[-1, :], z_f=z_all[-1, :], t_switch=t_switch)

    return x_all, y_all, z_all, u_all, j_f