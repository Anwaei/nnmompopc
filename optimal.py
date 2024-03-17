import numpy as np
from scipy.optimize import minimize
from scipy.special import legendre
import simulate
import dynamics2 as dyn
import simulate as simu
from config_opc import *
import plot_utils as pu
import casadi

DIM_CONS = PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +PARA_N_LGL_ALL*PARA_NU_AUXILIARY

def calculate_LGL_points(N_LGL):
    L = legendre(N_LGL-1)
    Ld = np.poly1d([i * L[i] for i in range(N_LGL-1, 0, -1)])
    LGL_points = np.append(Ld.r, [-1, 1])
    LGL_points.sort()    
    return LGL_points

def calculate_LGL_indexes(LGL_points, t_switch):
    LGL_time_1 = t_switch/2*LGL_points + t_switch/2
    LGL_time_2 = (PARA_TF-t_switch)/2*LGL_points + (PARA_TF+t_switch)/2
    LGL_time = np.concatenate((LGL_time_1, LGL_time_2))
    LGL_indexes_float = LGL_time/PARA_DT
    LGL_indexes = LGL_indexes_float.astype(int).tolist()    
    return LGL_indexes, LGL_time

def calculate_differential_matrix(N_LGL):
    LGL_points = calculate_LGL_points(N_LGL)
    # L = legendre(N_LGL-1)
    # legendre_values = L(LGL_points)
    D = np.zeros(shape=(N_LGL, N_LGL))
    # for m in range(N_LGL):
    #     for j in range(N_LGL):
    #         if m != j:
    #             D[m, j] = legendre_values[m] / legendre_values[j] / (LGL_points[m]-LGL_points[j])
    #         elif m == 0:
    #             D[m, j] = -(N_LGL-1)*N_LGL/4
    #         elif m == N_LGL-1:
    #             D[m, j] = (N_LGL-1)*N_LGL/4
    #         else:
    #             D[m, j] = 0
    for i in range(N_LGL):
        for k in range(N_LGL):
            if i == k:
                for j in range(N_LGL):
                    if j != i:
                        D[k, i] += 1/(LGL_points[i]-LGL_points[j])
            else:
                num = 1
                for j in range(N_LGL):
                    if j != i and j != k:
                        num *= LGL_points[k] - LGL_points[j]
                den = 1
                for j in range(N_LGL):
                    if j != i :
                        den *= LGL_points[i] - LGL_points[j]
                D[k, i] = num/den
    return D

def calculate_LGL_weights(LGL_points, tau, j):
    weight = 1
    for i in range(PARA_N_LGL_AGGRE):
        if i != j:
            weight *= (tau-LGL_points[i])/(LGL_points[j]-LGL_points[i])
    return weight

def interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch):
    LGL_points = calculate_LGL_points(PARA_N_LGL_AGGRE)
    LGL_indexex, LGL_time = calculate_LGL_indexes(LGL_points, t_switch)
    t = PARA_DT * np.arange(PARA_STEP_NUM)
    x = np.zeros((PARA_STEP_NUM, PARA_NX_AUXILIARY))
    y = np.zeros((PARA_STEP_NUM, PARA_NY_AUXILIARY))
    z = np.zeros((PARA_STEP_NUM, PARA_NZ_AUXILIARY))
    u = np.zeros((PARA_STEP_NUM, PARA_NU_AUXILIARY))
    for k in range(PARA_STEP_NUM):
        if t[k] < t_switch:
            tau = 2/(t_switch)*t[k] - 1
            for j in range(PARA_N_LGL_AGGRE):
                w = calculate_LGL_weights(LGL_points, tau=tau, j=j)
                x[k, :] = x[k, :] + w * x_optimal[j, :]
                y[k, :] = y[k, :] + w * y_optimal[j, :]
                z[k, :] = z[k, :] + w * z_optimal[j, :]
                u[k, :] = u[k, :] + w * u_optimal[j, :]
        elif t[k] > t_switch:
            tau = 2/(PARA_TF-t_switch)*t[k] - (PARA_TF+t_switch)/(PARA_TF-t_switch)
            for j in range(PARA_N_LGL_AGGRE, PARA_N_LGL_AGGRE+PARA_N_LGL_CRUISE):
                w = calculate_LGL_weights(LGL_points, tau=tau, j=j-PARA_N_LGL_AGGRE)
                x[k, :] = x[k, :] + w * x_optimal[j, :]
                y[k, :] = y[k, :] + w * y_optimal[j, :]
                z[k, :] = z[k, :] + w * z_optimal[j, :]
                u[k, :] = u[k, :] + w * u_optimal[j, :]
        else:
            res = np.concatenate((np.abs(x_optimal[PARA_N_LGL_AGGRE-1, :] - x_optimal[PARA_N_LGL_AGGRE, :]),
                          # np.abs(y_optimal[PARA_N_LGL_AGGRE-1, :] - y_optimal[PARA_N_LGL_AGGRE, :]),
                          np.abs(z_optimal[PARA_N_LGL_AGGRE-1, :] - z_optimal[PARA_N_LGL_AGGRE, :]),
                          np.abs(u_optimal[PARA_N_LGL_AGGRE-1, :] - u_optimal[PARA_N_LGL_AGGRE, :])), axis=-1)
            res_max = np.max(res)
            if res_max > 0.1:
                raise ValueError("optimal LGL results link error")
            x[k, :] = x_optimal[PARA_N_LGL_AGGRE, :]
            y[k, :] = y_optimal[PARA_N_LGL_AGGRE, :]
            z[k, :] = z_optimal[PARA_N_LGL_AGGRE, :]
            u[k, :] = u_optimal[PARA_N_LGL_AGGRE, :]
        
    return t, x, y, z, u

def function_objective(X, t_switch):
    y_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-PARA_NY_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-PARA_NZ_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)]
    gy = y_last - np.array([PARA_EPI12 * t_switch, PARA_EPI22 * (PARA_TF - t_switch)])
    max_gy = np.max(gy)
    cost = np.max((max_gy, -z_last[0]))
    return cost 

def print_C(x):
    print(x)

def function_constraint(X, t_switch, h_ref_lgl, diff_mat, x0):
    # t_switch = arg[0]
    # h_ref_lgl = arg[1]
    # diff_mat = arg[2]

    eq_cons_array = np.zeros(shape=PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
                                   +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY)
                                   )

    x_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]], (PARA_N_LGL_AGGRE, PARA_NX_AUXILIARY))  # Saved as row vector
    x_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]], (PARA_N_LGL_CRUISE, PARA_NX_AUXILIARY))
    y_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]], (PARA_N_LGL_AGGRE, PARA_NY_AUXILIARY))
    y_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]], (PARA_N_LGL_CRUISE, PARA_NY_AUXILIARY))
    z_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]], (PARA_N_LGL_AGGRE, PARA_NZ_AUXILIARY))
    z_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]], (PARA_N_LGL_CRUISE, PARA_NZ_AUXILIARY))
    u_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[7]], (PARA_N_LGL_AGGRE, PARA_NU_AUXILIARY))
    u_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[7]:PARA_INDEXES_VAR[8]], (PARA_N_LGL_CRUISE, PARA_NU_AUXILIARY))

    # constraints for x
    fx_matrix = np.zeros(shape=x_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fx_matrix[m, :] = dyn.dynamic_function(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    # fx_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]] = np.reshape(diff_mat @ x_aggre_matrix - fx_matrix, PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY)

    for m in range(PARA_N_LGL_CRUISE):
        fx_matrix[m, :] = dyn.dynamic_function(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    # fx_matrix *= (PARA_TF-t_switch)/2    
    eq_cons_array[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]] = np.reshape(diff_mat @ x_cruise_matrix - fx_matrix, PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY)
    
    # constraints for y
    fy_matrix = np.zeros(shape=y_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fy_matrix[m, :] = np.array((dyn.cost_origin_aggressive(x_aggre_matrix[m, :], u_aggre_matrix[m, :], h_ref_lgl[m])[1], 0))
    # fy_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]] = np.reshape(diff_mat @ y_aggre_matrix - fy_matrix, PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY)
    for m in range(PARA_N_LGL_CRUISE):
        fy_matrix[m, :] = np.array((0, dyn.cost_origin_cruise(x_cruise_matrix[m, :], u_cruise_matrix[m, :], h_ref_lgl[m+PARA_N_LGL_AGGRE])[1]))
    # fy_matrix *= (PARA_TF-t_switch)/2   
    eq_cons_array[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]] = np.reshape(diff_mat @ y_cruise_matrix - fy_matrix, PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY)

    # constraints for z
    fz_matrix = np.zeros(shape=z_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fz_matrix[m, :] = dyn.cost_tracking_error(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    # fz_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]] = np.reshape(diff_mat @ z_aggre_matrix + fz_matrix, PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY)
    for m in range(PARA_N_LGL_CRUISE):
        fz_matrix[m, :] = dyn.cost_tracking_error(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    # fz_matrix *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]] = np.reshape(diff_mat @ z_cruise_matrix + fz_matrix, PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY)

    eq_cons_array[PARA_INDEXES_VAR[6]:] = np.concatenate((x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], y_aggre_matrix[-1,:]-y_cruise_matrix[0, :],
                                                            z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :]))

    return eq_cons_array

def generate_initial_variables(x0, traject_ref, method="pid"):
    x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, aero_info = simu.simulate_auxiliary(x0=x0, trajectory_ref=traject_ref, control_method=method)
    # y_all_aux = np.zeros(y_all_aux.shape)
    # pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, traject_ref, aero_info)
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=traject_ref['t_switch'])
    X = zip_variable(x_all_aux[LGL_indexes, :], y_all_aux[LGL_indexes, :], z_all_aux[LGL_indexes, :], u_all_aux[LGL_indexes, :])

    return X


def zip_variable(x_matrix, y_matrix, z_matrix, u_matrix):
    X = np.zeros(shape=PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY))
    X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[2]] = np.reshape(x_matrix, PARA_N_LGL_ALL*PARA_NX_AUXILIARY)
    X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[4]] = np.reshape(y_matrix, PARA_N_LGL_ALL*PARA_NY_AUXILIARY)
    X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[6]] = np.reshape(z_matrix, PARA_N_LGL_ALL*PARA_NZ_AUXILIARY)
    X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]] = np.reshape(u_matrix, PARA_N_LGL_ALL*PARA_NU_AUXILIARY)
    return X

def unzip_variable(X):
    x_matrix = np.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[2]], (PARA_N_LGL_ALL, PARA_NX_AUXILIARY))  # Saved as row vector
    y_matrix = np.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[4]], (PARA_N_LGL_ALL, PARA_NY_AUXILIARY))
    z_matrix = np.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[6]], (PARA_N_LGL_ALL, PARA_NZ_AUXILIARY))
    u_matrix = np.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]], (PARA_N_LGL_ALL, PARA_NU_AUXILIARY))
    return x_matrix, y_matrix, z_matrix, u_matrix


def generate_PS_solution(x0, trajectory_ref):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)  # Assume same number
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=t_switch)
    h_ref_lgl = h_ref[LGL_indexes]
    diff_mat = calculate_differential_matrix(PARA_N_LGL_AGGRE)  # Assume same number
    X0 = generate_initial_variables(x0=PARA_X0, traject_ref=trajectory_ref, method="pid")

    fun_obj = function_objective
    fun_cons = function_constraint
    constraints = {'type': 'eq', 'fun': fun_cons, 'args': (t_switch, h_ref_lgl, diff_mat, PARA_X0)} 
    optimal_solution = minimize(fun=fun_obj, x0=X0, constraints=constraints, args=t_switch, method='trust-constr')
    x_optimal, y_optimal, z_optimal, u_optimal = unzip_variable(optimal_solution.x)
    j_optimal = optimal_solution.fun

    return x_optimal, y_optimal, z_optimal, u_optimal, j_optimal


def function_objective_casadi(X, t_switch):
    y_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-PARA_NY_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-PARA_NZ_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)]
    # z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    # gy = y_last - casadi.MX([PARA_EPI12 * t_switch, PARA_EPI22 * (PARA_TF - t_switch)])
    # max_gy = casadi.mmax(gy)
    # cost = casadi.fmax(max_gy, -z_last[0])
    return -z_last[0]
    # cost = casadi.fmin(max_gy+50, -z_last[0])
    # return cost

def function_constraint_casadi(X, t_switch, h_ref_lgl, diff_mat, x0):
    # t_switch = arg[0]
    # h_ref_lgl = arg[1]
    # diff_mat = arg[2]

    eq_cons_array = casadi.MX.zeros(DIM_CONS)

    x_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]], (PARA_NX_AUXILIARY, PARA_N_LGL_AGGRE)))  # Saved as row vector
    x_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]], (PARA_NX_AUXILIARY, PARA_N_LGL_CRUISE)))
    y_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]], (PARA_NY_AUXILIARY, PARA_N_LGL_AGGRE)))
    y_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]], (PARA_NY_AUXILIARY, PARA_N_LGL_CRUISE)))
    z_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]], (PARA_NZ_AUXILIARY, PARA_N_LGL_AGGRE)))
    z_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]], (PARA_NZ_AUXILIARY, PARA_N_LGL_CRUISE)))
    u_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[7]], (PARA_NU_AUXILIARY, PARA_N_LGL_AGGRE)))
    u_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[7]:PARA_INDEXES_VAR[8]], (PARA_NU_AUXILIARY, PARA_N_LGL_CRUISE)))

    # constraints for x
    fx_matrix = casadi.MX.zeros(x_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fx_matrix[m, :] = dyn.dynamic_function_casadi(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    fx_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]] = casadi.reshape(diff_mat @ x_aggre_matrix - fx_matrix, (PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY, 1))
    
    fx_matrix = casadi.MX.zeros(x_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fx_matrix[m, :] = dyn.dynamic_function_casadi(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    fx_matrix *= (PARA_TF-t_switch)/2    
    eq_cons_array[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]] = casadi.reshape(diff_mat @ x_cruise_matrix - fx_matrix, (PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY, 1))
    
    # constraints for y
    # V: x[0] T: u[1]
    # fy_matrix = casadi.MX.zeros(y_aggre_matrix.shape)
    # for m in range(PARA_N_LGL_AGGRE):
    #     fy_matrix[m, :] = casadi.vertcat(x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM, 0)
    #     # fy_matrix[m, :] = casadi.vertcat(dyn.cost_origin_cruise_casadi(x_aggre_matrix[m, :], u_aggre_matrix[m, :], h_ref_lgl[m])[1], 0)
    # fy_matrix *= t_switch/2
    # eq_cons_array[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]] = casadi.reshape(diff_mat @ y_aggre_matrix - fy_matrix, (PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY, 1))
    # # eq_cons_array[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]] = casadi.reshape(y_aggre_matrix, (PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY, 1))
    #
    # fy_matrix = casadi.MX.zeros(y_cruise_matrix.shape)
    # for m in range(PARA_N_LGL_CRUISE):
    #     # fy_matrix[m, :] = casadi.vertcat(0, dyn.cost_origin_cruise_casadi(x_cruise_matrix[m, :], u_cruise_matrix[m, :], h_ref_lgl[m+PARA_N_LGL_AGGRE])[1])
    #     fy_matrix[m, :] = casadi.vertcat(0, x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM)
    # fy_matrix *= (PARA_TF-t_switch)/2
    # eq_cons_array[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]] = casadi.reshape(diff_mat @ y_cruise_matrix - fy_matrix, (PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY, 1))
    # eq_cons_array[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[3]+2] = 0
    # # eq_cons_array[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]] = casadi.reshape(y_cruise_matrix[:, :],
    # #                                                                         ((PARA_N_LGL_CRUISE) * PARA_NY_AUXILIARY, 1))
    # # eq_cons_array[PARA_INDEXES_VAR[3]+2:PARA_INDEXES_VAR[4]] = 0

    # fy_matrix_1 = casadi.MX.zeros((PARA_N_LGL_AGGRE, 1))
    # for m in range(PARA_N_LGL_AGGRE):
    #     fy_matrix_1[m] = x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM
    # fy_matrix_1 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[2]+PARA_N_LGL_AGGRE] = diff_mat @ y_aggre_matrix[:, 0] - fy_matrix_1
    # # eq_cons_array[PARA_INDEXES_VAR[2]+PARA_N_LGL_AGGRE:PARA_INDEXES_VAR[3]] = y_aggre_matrix[:, 1]
    #
    # fy_matrix_2 = casadi.MX.zeros((PARA_N_LGL_CRUISE, 1))
    # for m in range(PARA_N_LGL_CRUISE):
    #     fy_matrix_2[m] = x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM
    # fy_matrix_2 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[3]+PARA_N_LGL_AGGRE] = diff_mat @ y_cruise_matrix[:, 1] - fy_matrix_2
    # eq_cons_array[PARA_INDEXES_VAR[3]+PARA_N_LGL_AGGRE:PARA_INDEXES_VAR[4]] = y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0]


    # constraints for z
    fz_matrix = casadi.MX.zeros(z_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fz_matrix[m, 0] = -dyn.cost_tracking_error(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    fz_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]] = casadi.reshape(diff_mat @ z_aggre_matrix - fz_matrix, (PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY, 1))
    
    fz_matrix = casadi.MX.zeros(z_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fz_matrix[m, 0] = -dyn.cost_tracking_error(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    fz_matrix *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]] = casadi.reshape(diff_mat @ z_cruise_matrix - fz_matrix, (PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY, 1))

    # constraints for link
    link_index = PARA_INDEXES_VAR[6]+(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY)
    eq_cons_array[PARA_INDEXES_VAR[6]:link_index] = casadi.horzcat(x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], y_aggre_matrix[-1,:]-y_cruise_matrix[0, :],
                                                            z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :])
    # eq_cons_array[PARA_INDEXES_VAR[6]+PARA_NX_AUXILIARY:PARA_INDEXES_VAR[6]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY] = 0
    # eq_cons_array[PARA_INDEXES_VAR[6]:link_index] = casadi.horzcat(x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], casadi.MX.zeros(1,2),
    #                                                         z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :])


    # constraints for start value
    eq_cons_array[link_index:link_index+PARA_NX_AUXILIARY] = x_aggre_matrix[0,:] - casadi.reshape(x0, (1, PARA_NX_AUXILIARY))
    # eq_cons_array[link_index+PARA_NX_AUXILIARY:link_index+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY] = y_aggre_matrix[0, :]
    # eq_cons_array[link_index+PARA_NX_AUXILIARY] = y_aggre_matrix[0,0]
    # eq_cons_array[link_index+PARA_NX_AUXILIARY+1] = y_cruise_matrix[0,1]
    eq_cons_array[link_index+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY:link_index+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY] = z_aggre_matrix[0,:]
    start_index = link_index + PARA_NX_AUXILIARY + PARA_NY_AUXILIARY + PARA_NZ_AUXILIARY

    # constraints for control input
    for m in range(PARA_N_LGL_AGGRE):
        eq_cons_array[start_index + m*PARA_NU_AUXILIARY: start_index + (m+1)*PARA_NU_AUXILIARY] = u_aggre_matrix[m, :]
    start_index2 = start_index + PARA_N_LGL_AGGRE*PARA_NU_AUXILIARY
    for m in range(PARA_N_LGL_CRUISE):
        eq_cons_array[start_index2 + m*PARA_NU_AUXILIARY: start_index2 + (m+1)*PARA_NU_AUXILIARY] = u_cruise_matrix[m, :]

    # return casadi.vertcat(eq_cons_array[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[2]], eq_cons_array[PARA_INDEXES_VAR[4]:])
    return eq_cons_array
    # return eq_cons_array


def generate_PS_solution_casadi(x0, trajectory_ref):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)  # Assume same number
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=t_switch)
    h_ref_lgl = h_ref[LGL_indexes]
    diff_mat = calculate_differential_matrix(PARA_N_LGL_AGGRE)  # Assume same number
    X0 = generate_initial_variables(x0=PARA_X0, traject_ref=trajectory_ref, method="pid")

    V = casadi.MX.sym("V", PARA_INDEXES_VAR[-1])
    J = function_objective_casadi(X=V, t_switch=t_switch)
    g = function_constraint_casadi(X=V, t_switch=t_switch, h_ref_lgl=h_ref_lgl, diff_mat=diff_mat, x0=PARA_X0)
    # lbg = np.zeros(PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
    #                                +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY))
    # ubg = np.zeros(PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
    #                                +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY))
    dim_constraints = DIM_CONS
    lbg = np.zeros(dim_constraints)
    ubg = np.zeros(dim_constraints)
    lbg[-PARA_N_LGL_ALL*PARA_NU_AUXILIARY:] = np.concatenate([PARA_U_LOWER_BOUND for i in range(PARA_N_LGL_ALL)], 0)
    ubg[-PARA_N_LGL_ALL*PARA_NU_AUXILIARY:] = np.concatenate([PARA_U_UPPER_BOUND for i in range(PARA_N_LGL_ALL)], 0)

    nlp = {'x':V, 'f':J, 'g':g}
    opts = {}
    opts["expand"] = True
    # opts["ipopt.max_iter"] = 4
    # opts["ipopt.linear_solver"] = 'ma27'
    solver = casadi.nlpsol('S', 'ipopt', nlp, opts)
    res = solver(x0=X0, lbg=lbg, ubg=ubg)
    print("optimal cost: ", float(res["f"]))
    v_opt = np.array(res["x"])
    x_optimal, y_optimal, z_optimal, u_optimal = unzip_variable(v_opt)
    j_optimal = np.array(res["f"])

    # print("============================== Opti results =============================")
    # opti = casadi.Opti()
    # xo = opti.variable(PARA_INDEXES_VAR[-1])
    # opti.minimize(function_objective_casadi(X=xo, t_switch=t_switch))
    # opti.subject_to(function_constraint_casadi(X=xo, t_switch=t_switch, h_ref_lgl=h_ref_lgl, diff_mat=diff_mat, x0=PARA_X0) == 0)
    # opti.solver('ipopt')
    # sol = opti.solve()

    return x_optimal, y_optimal, z_optimal, u_optimal, j_optimal

