import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import legendre
import simulate
import dynamics2 as dyn
import simulate as simu
from config_opc import *
import plot_utils as pu
import casadi

# DIM_CONS = PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
#            +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
#            +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
#            +PARA_N_LGL_ALL*PARA_NU_AUXILIARY \
#            +PARA_N_LGL_ALL*1

DIM_CONS = PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
           +PARA_N_LGL_ALL*PARA_NU_AUXILIARY \

DIM_CONS_LGR = (PARA_N_LGL_ALL-1)*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +PARA_N_LGL_ALL*PARA_NU_AUXILIARY

X0 = PARA_X0
U0 = PARA_U0

X0_SCALED = X0.copy()
X0_SCALED[4] = (X0_SCALED[4]-SCALE_MEAN_H)/SCALE_VAR_H
X0_SCALED[0] = (X0_SCALED[0]-SCALE_MEAN_V)/SCALE_VAR_V
U0_SCALED = U0.copy()
U0_SCALED[1] = (U0_SCALED[1]-SCALE_MEAN_T)/SCALE_VAR_T

def scale_init(x_init, u_init):
    x_init_scaled = x_init.copy()
    u_init_scaled = u_init.copy()
    x_init_scaled[:, 4] = (x_init[:, 4] - SCALE_MEAN_H)/SCALE_VAR_H
    x_init_scaled[:, 0] = (x_init[:, 0] - SCALE_MEAN_V)/SCALE_VAR_V
    u_init_scaled[:, 1] = (u_init[:, 1] - SCALE_MEAN_T)/SCALE_VAR_T
    return x_init_scaled, u_init_scaled

def weighted_exp_loss(x):
    return (PARA_WP*casadi.exp(x) + PARA_WN*casadi.exp(-x))/PARA_ELOSSSCALE

def smooth_loss(u_set):
    loss = 0
    for k in range(1, PARA_N_LGL_AGGRE-1):
        loss += weighted_exp_loss((u_set[k]-u_set[k-1])*(u_set[k+1]-u_set[k]))
    return loss

# def second_derivative_loss(u_set):

def ref_xi_loss(u_set, ref_xi):
    return casadi.sum1((u_set-ref_xi)**2)/PARA_XILOSSSCALE

def calculate_LGL_points(N_LGL):
    L = legendre(N_LGL-1)
    Ld = np.poly1d([i * L[i] for i in range(N_LGL-1, 0, -1)])
    LGL_points = np.append(Ld.r, [-1, 1])
    LGL_points.sort()    
    return LGL_points

def calculate_LGR_points(N_LGR):
    L = legendre(N_LGR-1) + legendre(N_LGR-2)
    LGR_points = np.append(L.r, [1])
    LGR_points.sort()    
    return LGR_points

def calculate_LGL_indexes(LGL_points, t_switch):
    LGL_time_1 = t_switch/2*LGL_points + t_switch/2
    LGL_time_2 = (PARA_TF-t_switch)/2*LGL_points + (PARA_TF+t_switch)/2
    LGL_time = np.concatenate((LGL_time_1, LGL_time_2))
    LGL_indexes_float = LGL_time/PARA_DT
    LGL_indexes = LGL_indexes_float.astype(int).tolist()    
    return LGL_indexes, LGL_time

def calculate_LGR_indexes(LGR_points, t_switch):
    LGR_time_1 = t_switch/2*LGR_points + t_switch/2
    LGR_time_2 = (PARA_TF-t_switch)/2*LGR_points + (PARA_TF+t_switch)/2
    LGR_time = np.concatenate((LGR_time_1, LGR_time_2))
    LGR_indexes_float = LGR_time/PARA_DT
    LGR_indexes = LGR_indexes_float.astype(int).tolist()    
    return LGR_indexes, LGR_time

def calculate_differential_matrix_LGL(N_LGL):
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

def calculate_differential_matrix_LGR(N_LGR):
    LGR_points = calculate_LGR_points(N_LGR)
    D = np.zeros(shape=(N_LGR-1, N_LGR))
    for i in range(N_LGR):
        for k in range(N_LGR-1):
            if i == k:
                for j in range(N_LGR):
                    if j != i:
                        D[k, i] += 1/(LGR_points[i]-LGR_points[j])
            else:
                num = 1
                for j in range(N_LGR):
                    if j != i and j != k:
                        num *= LGR_points[k] - LGR_points[j]
                den = 1
                for j in range(N_LGR):
                    if j != i :
                        den *= LGR_points[i] - LGR_points[j]
                D[k, i] = num/den
    return D

def calculate_LGL_weights(LGL_points, tau, j):
    weight = 1
    for i in range(PARA_N_LGL_AGGRE):
        if i != j:
            weight *= (tau-LGL_points[i])/(LGL_points[j]-LGL_points[i])
    return weight

def calculate_LGR_weights(LGR_points, tau, j):
    weight = 1
    for i in range(PARA_N_LGL_AGGRE):
        if i != j:
            weight *= (tau-LGR_points[i])/(LGR_points[j]-LGR_points[i])
    return weight

def interpolate_optimal_trajectory(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch, from_scaled=False):
    if from_scaled:
        x_optimal[:, 4] = dyn.re_state(x_optimal[:, 4], SCALE_MEAN_H, SCALE_VAR_H)
        x_optimal[:, 0] = dyn.re_state(x_optimal[:, 0], SCALE_MEAN_V, SCALE_VAR_V)
        u_optimal[:, 1] = dyn.re_state(u_optimal[:, 1], SCALE_MEAN_T, SCALE_VAR_T)

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

def interpolate_optimal_trajectory_LGR(x_optimal, y_optimal, z_optimal, u_optimal, j_optimal, t_switch):
    LGR_points = calculate_LGR_points(PARA_N_LGL_AGGRE)
    LGR_indexex, LGR_time = calculate_LGR_indexes(LGR_points, t_switch)
    t = PARA_DT * np.arange(PARA_STEP_NUM)
    x = np.zeros((PARA_STEP_NUM, PARA_NX_AUXILIARY))
    y = np.zeros((PARA_STEP_NUM, PARA_NY_AUXILIARY))
    z = np.zeros((PARA_STEP_NUM, PARA_NZ_AUXILIARY))
    u = np.zeros((PARA_STEP_NUM, PARA_NU_AUXILIARY))
    for k in range(PARA_STEP_NUM):
        if t[k] < t_switch:
            tau = 2/(t_switch)*t[k] - 1
            for j in range(PARA_N_LGL_AGGRE):
                w = calculate_LGR_weights(LGR_points, tau=tau, j=j)
                x[k, :] = x[k, :] + w * x_optimal[j, :]
                y[k, :] = y[k, :] + w * y_optimal[j, :]
                z[k, :] = z[k, :] + w * z_optimal[j, :]
                u[k, :] = u[k, :] + w * u_optimal[j, :]
        elif t[k] > t_switch:
            tau = 2/(PARA_TF-t_switch)*t[k] - (PARA_TF+t_switch)/(PARA_TF-t_switch)
            for j in range(PARA_N_LGL_AGGRE, PARA_N_LGL_AGGRE+PARA_N_LGL_CRUISE):
                w = calculate_LGR_weights(LGR_points, tau=tau, j=j-PARA_N_LGL_AGGRE)
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
                raise ValueError("optimal LGR results link error")
            x[k, :] = x_optimal[PARA_N_LGL_AGGRE, :]
            y[k, :] = y_optimal[PARA_N_LGL_AGGRE, :]
            z[k, :] = z_optimal[PARA_N_LGL_AGGRE, :]
            u[k, :] = u_optimal[PARA_N_LGL_AGGRE, :]
        
    return t, x, y, z, u

def function_objective(X, t_switch):
    # y_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-PARA_NY_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)]
    # z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-PARA_NZ_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)]
    # gy = y_last - np.array([PARA_EPI12 * t_switch, PARA_EPI22 * (PARA_TF - t_switch)])
    # max_gy = np.max(gy)
    # cost = np.max((max_gy, -z_last[0]))
    # return cost
    return X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]

def function_objective_fuel(X, t_switch):
    y_last_aggre = X[PARA_N_LGL_ALL * PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE * PARA_NY_AUXILIARY - 2]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    gy = y_last_aggre - PARA_EPI12 * t_switch
    cost = np.max((gy, z_last))
    return cost

def print_C(x):
    print(x)

DIM_CONS_NP_LGL =  PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
           +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)

def function_constraint(X, t_switch, h_ref_lgl, diff_mat, x0):
    # t_switch = arg[0]
    # h_ref_lgl = arg[1]
    # diff_mat = arg[2]

    eq_cons_array = np.ones(DIM_CONS_NP_LGL)

    # eq_cons_array = np.zeros(shape=PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
    #                             +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
    #                             +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
    #                                )

    x_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]], (PARA_N_LGL_AGGRE, PARA_NX_AUXILIARY))  # Saved as row vector
    x_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]], (PARA_N_LGL_CRUISE, PARA_NX_AUXILIARY))
    y_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]], (PARA_N_LGL_AGGRE, PARA_NY_AUXILIARY))
    y_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]], (PARA_N_LGL_CRUISE, PARA_NY_AUXILIARY))
    z_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]], (PARA_N_LGL_AGGRE, PARA_NZ_AUXILIARY))
    z_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]], (PARA_N_LGL_CRUISE, PARA_NZ_AUXILIARY))
    u_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[7]], (PARA_N_LGL_AGGRE, PARA_NU_AUXILIARY))
    u_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[7]:PARA_INDEXES_VAR[8]], (PARA_N_LGL_CRUISE, PARA_NU_AUXILIARY))

    # constraints for x
    fx_matrix1 = np.zeros(shape=x_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fx_matrix1[m, :] = dyn.dynamic_function(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    fx_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[0]:PARA_INDEXES_CONS_NP_LGL[1]] = np.reshape(diff_mat @ x_aggre_matrix - fx_matrix1, PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY)

    fx_matrix2 = np.zeros(shape=x_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fx_matrix2[m, :] = dyn.dynamic_function(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    fx_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[1]:PARA_INDEXES_CONS_NP_LGL[2]] = np.reshape(diff_mat @ x_cruise_matrix - fx_matrix2, PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY)
    
    # # constraints for y
    # fy_matrix1 = np.zeros(shape=y_aggre_matrix.shape)
    # for m in range(PARA_N_LGL_AGGRE):
    #     fy_matrix1[m, :] = np.array((x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM, 0))
    # fy_matrix1 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_CONS_NP_LGL[2]:PARA_INDEXES_CONS_NP_LGL[3]] = np.reshape(diff_mat @ y_aggre_matrix - fy_matrix1, PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY)

    # fy_matrix2 = np.zeros(shape=y_cruise_matrix.shape)
    # for m in range(PARA_N_LGL_CRUISE):
    #     fy_matrix2[m, :] = np.array((0, x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM)) # Not modified
    # fy_matrix2 *= (PARA_TF-t_switch)/2
    # eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]:PARA_INDEXES_CONS_NP_LGL[4]] = np.reshape(diff_mat @ y_cruise_matrix - fy_matrix2, PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY)

    # constraints for y
    fy_matrix1 = np.zeros(shape=(PARA_N_LGL_AGGRE, 1))
    for m in range(PARA_N_LGL_AGGRE):
        fy_matrix1[m, :] = x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM
    fy_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[2]:PARA_INDEXES_CONS_NP_LGL[3]-PARA_N_LGL_AGGRE] = np.reshape(diff_mat @ y_aggre_matrix[:, 0, np.newaxis] - fy_matrix1, PARA_N_LGL_AGGRE)
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]-PARA_N_LGL_AGGRE:PARA_INDEXES_CONS_NP_LGL[3]] = y_aggre_matrix[:, 1]


    fy_matrix2 = np.zeros(shape=(PARA_N_LGL_AGGRE, 1))
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]:PARA_INDEXES_CONS_NP_LGL[4]-PARA_N_LGL_AGGRE] = (y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0])
    for m in range(PARA_N_LGL_CRUISE):
        fy_matrix2[m, :] = x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM # Not modified
    fy_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[4]-PARA_N_LGL_AGGRE:PARA_INDEXES_CONS_NP_LGL[4]] = np.reshape(diff_mat @ y_cruise_matrix[:, 1, np.newaxis] - fy_matrix2, PARA_N_LGL_CRUISE)              

    # constraints for z
    fz_matrix1 = np.zeros(shape=z_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fz_matrix1[m, :] = dyn.cost_tracking_error(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    fz_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[4]:PARA_INDEXES_CONS_NP_LGL[5]] = np.reshape(diff_mat @ z_aggre_matrix - fz_matrix1, PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY)

    fz_matrix2 = np.zeros(shape=z_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fz_matrix2[m, :] = dyn.cost_tracking_error(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    fz_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[5]:PARA_INDEXES_CONS_NP_LGL[6]] = np.reshape(diff_mat @ z_cruise_matrix - fz_matrix2, PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY)

    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[6]:PARA_INDEXES_CONS_NP_LGL[7]] = np.concatenate((x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :]))

    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[7]:PARA_INDEXES_CONS_NP_LGL[8]] = np.concatenate((x_aggre_matrix[0, :]-x0, y_aggre_matrix[0, 0, np.newaxis], y_cruise_matrix[0, 1, np.newaxis], z_aggre_matrix[0, :]))



    return eq_cons_array


def function_constraint_scaled(X, t_switch, h_ref_lgl, diff_mat, x0):
    # t_switch = arg[0]
    # h_ref_lgl = arg[1]
    # diff_mat = arg[2]

    eq_cons_array = np.ones(DIM_CONS_NP_LGL)

    # eq_cons_array = np.zeros(shape=PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
    #                             +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
    #                             +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
    #                                )

    x_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]], (PARA_N_LGL_AGGRE, PARA_NX_AUXILIARY))  # Saved as row vector
    x_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]], (PARA_N_LGL_CRUISE, PARA_NX_AUXILIARY))
    y_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]], (PARA_N_LGL_AGGRE, PARA_NY_AUXILIARY))
    y_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]], (PARA_N_LGL_CRUISE, PARA_NY_AUXILIARY))
    z_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]], (PARA_N_LGL_AGGRE, PARA_NZ_AUXILIARY))
    z_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]], (PARA_N_LGL_CRUISE, PARA_NZ_AUXILIARY))
    u_aggre_matrix = np.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[7]], (PARA_N_LGL_AGGRE, PARA_NU_AUXILIARY))
    u_cruise_matrix = np.reshape(X[PARA_INDEXES_VAR[7]:PARA_INDEXES_VAR[8]], (PARA_N_LGL_CRUISE, PARA_NU_AUXILIARY))

    # constraints for x
    fx_matrix1 = np.zeros(shape=x_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fx_matrix1[m, :] = dyn.dynamic_function_scaled(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    fx_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[0]:PARA_INDEXES_CONS_NP_LGL[1]] = np.reshape(diff_mat @ x_aggre_matrix - fx_matrix1, PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY)

    fx_matrix2 = np.zeros(shape=x_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fx_matrix2[m, :] = dyn.dynamic_function_scaled(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    fx_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[1]:PARA_INDEXES_CONS_NP_LGL[2]] = np.reshape(diff_mat @ x_cruise_matrix - fx_matrix2, PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY)
    
    # # constraints for y
    # fy_matrix1 = np.zeros(shape=y_aggre_matrix.shape)
    # for m in range(PARA_N_LGL_AGGRE):
    #     fy_matrix1[m, :] = np.array((x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM, 0))
    # fy_matrix1 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_CONS_NP_LGL[2]:PARA_INDEXES_CONS_NP_LGL[3]] = np.reshape(diff_mat @ y_aggre_matrix - fy_matrix1, PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY)

    # fy_matrix2 = np.zeros(shape=y_cruise_matrix.shape)
    # for m in range(PARA_N_LGL_CRUISE):
    #     fy_matrix2[m, :] = np.array((0, x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM)) # Not modified
    # fy_matrix2 *= (PARA_TF-t_switch)/2
    # eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]:PARA_INDEXES_CONS_NP_LGL[4]] = np.reshape(diff_mat @ y_cruise_matrix - fy_matrix2, PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY)

    # constraints for y
    fy_matrix1 = np.zeros(shape=(PARA_N_LGL_AGGRE, 1))
    for m in range(PARA_N_LGL_AGGRE):
        fy_matrix1[m, :] = x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM
    fy_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[2]:PARA_INDEXES_CONS_NP_LGL[3]-PARA_N_LGL_AGGRE] = np.reshape(diff_mat @ y_aggre_matrix[:, 0, np.newaxis] - fy_matrix1, PARA_N_LGL_AGGRE)
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]-PARA_N_LGL_AGGRE:PARA_INDEXES_CONS_NP_LGL[3]] = y_aggre_matrix[:, 1]


    fy_matrix2 = np.zeros(shape=(PARA_N_LGL_AGGRE, 1))
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[3]:PARA_INDEXES_CONS_NP_LGL[4]-PARA_N_LGL_AGGRE] = (y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0])
    for m in range(PARA_N_LGL_CRUISE):
        fy_matrix2[m, :] = x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM # Not modified
    fy_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[4]-PARA_N_LGL_AGGRE:PARA_INDEXES_CONS_NP_LGL[4]] = np.reshape(diff_mat @ y_cruise_matrix[:, 1, np.newaxis] - fy_matrix2, PARA_N_LGL_CRUISE)

    # constraints for z
    fz_matrix1 = np.zeros(shape=z_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fz_matrix1[m, :] = dyn.cost_tracking_error_scaled(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    fz_matrix1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[4]:PARA_INDEXES_CONS_NP_LGL[5]] = np.reshape(diff_mat @ z_aggre_matrix - fz_matrix1, PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY)

    fz_matrix2 = np.zeros(shape=z_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fz_matrix2[m, :] = dyn.cost_tracking_error_scaled(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    fz_matrix2 *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[5]:PARA_INDEXES_CONS_NP_LGL[6]] = np.reshape(diff_mat @ z_cruise_matrix - fz_matrix2, PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY)

    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[6]:PARA_INDEXES_CONS_NP_LGL[7]] = np.concatenate((x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :]))

    eq_cons_array[PARA_INDEXES_CONS_NP_LGL[7]:PARA_INDEXES_CONS_NP_LGL[8]] = np.concatenate((x_aggre_matrix[0, :]-x0, y_aggre_matrix[0, 0, np.newaxis], y_cruise_matrix[0, 1, np.newaxis], z_aggre_matrix[0, :]))

    return eq_cons_array

 

def generate_initial_variables(x0, traject_ref, method="pid"):
    x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, aero_info = simu.simulate_auxiliary(x0=x0, trajectory_ref=traject_ref, control_method=method)
    # y_all_aux = np.zeros(y_all_aux.shape)
    # pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, traject_ref, aero_info)
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=traject_ref['t_switch'])
    X = zip_variable(x_all_aux[LGL_indexes, :], y_all_aux[LGL_indexes, :], z_all_aux[LGL_indexes, :], u_all_aux[LGL_indexes, :])

    return X

def generate_initial_variables_scaled(x0, traject_ref, method="pid"):
    x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, aero_info = simu.simulate_auxiliary(x0=x0, trajectory_ref=traject_ref, control_method=method)
    # y_all_aux = np.zeros(y_all_aux.shape)
    # pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, traject_ref, aero_info)
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=traject_ref['t_switch'])
    x_init_scaled, u_init_scaled = scale_init(x_all_aux, u_all_aux)
    X = zip_variable(x_init_scaled[LGL_indexes, :], y_all_aux[LGL_indexes, :], z_all_aux[LGL_indexes, :], u_init_scaled[LGL_indexes, :])

    return X

def generate_initial_variables_LGR(x0, traject_ref, method="pid"):
    x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, aero_info = simu.simulate_auxiliary(x0=x0, trajectory_ref=traject_ref, control_method=method)
    # y_all_aux = np.zeros(y_all_aux.shape)
    # pu.plot_trajectory_auxiliary(x_all_aux, y_all_aux, z_all_aux, u_all_aux, j_f_aux, traject_ref, aero_info)
    LGR_points = calculate_LGR_points(N_LGR=PARA_N_LGL_AGGRE)
    LGR_indexes, _ = calculate_LGR_indexes(LGR_points=LGR_points, t_switch=traject_ref['t_switch'])
    X = zip_variable(x_all_aux[LGR_indexes, :], y_all_aux[LGR_indexes, :], z_all_aux[LGR_indexes, :], u_all_aux[LGR_indexes, :])

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


def callback_PS(x):
    # global ITER_NUM
    # cost = function_objective(x, PARA_TF*0.5)
    # print(f"Iteration {ITER_NUM}, cost: {cost}")
    # ITER_NUM += 1
    pass
    # return
ITER_NUM = 0

def generate_PS_solution(x0, trajectory_ref):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)  # Assume same number
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=t_switch)
    h_ref_lgl = h_ref[LGL_indexes]
    diff_mat = calculate_differential_matrix_LGL(PARA_N_LGL_AGGRE)  # Assume same number
    X0 = generate_initial_variables(x0=PARA_X0, traject_ref=trajectory_ref, method="pid")

    fun_obj = function_objective
    fun_cons = function_constraint
    constraints = {'type': 'eq', 'fun': fun_cons, 'args': (t_switch, h_ref_lgl, diff_mat, PARA_X0)}
    ulb = PARA_U_LOWER_BOUND.copy()
    uub = PARA_U_UPPER_BOUND.copy()    
    # if not (morphing_disabled is None):
    #     ulb[-1] = morphing_disabled
    #     uub[-1] = morphing_disabled
    lb = np.zeros(PARA_DIMENSION_VAR)
    ub = np.zeros(PARA_DIMENSION_VAR)
    lb[:PARA_INDEXES_VAR[6]] = -np.inf
    ub[:PARA_INDEXES_VAR[6]] = np.inf        
    lb[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]] = np.concatenate([ulb for i in range(PARA_N_LGL_ALL)])
    ub[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]] = np.concatenate([uub for i in range(PARA_N_LGL_ALL)])
    bounds = Bounds(lb=lb, ub=ub)
    optimal_solution = minimize(fun=fun_obj, x0=X0, constraints=constraints, bounds=bounds, args=t_switch, method='SLSQP', callback=callback_PS, options={'maxiter': 1000, 'disp': True, 'iprint': 2})
    x_optimal, y_optimal, z_optimal, u_optimal = unzip_variable(optimal_solution.x)
    j_optimal = optimal_solution.fun

    return x_optimal, y_optimal, z_optimal, u_optimal, j_optimal


def generate_PS_solution_scaled(x0, trajectory_ref, morphing_disabled=None, fun_obj = function_objective):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)  # Assume same number
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=t_switch)
    h_ref_lgl = h_ref[LGL_indexes]
    diff_mat = calculate_differential_matrix_LGL(PARA_N_LGL_AGGRE)  # Assume same number
    Xinitial_scaled = generate_initial_variables_scaled(x0=PARA_X0, traject_ref=trajectory_ref, method="pid")

    fun_obj_scaled = fun_obj
    fun_cons_scaled = function_constraint_scaled
    constraints = {'type': 'eq', 'fun': fun_cons_scaled, 'args': (t_switch, h_ref_lgl, diff_mat, X0_SCALED)}
    ulb = PARA_U_LOWER_BOUND.copy()
    uub = PARA_U_UPPER_BOUND.copy()
    ulb[1] = dyn.scale_state(ulb[1], SCALE_MEAN_T, SCALE_VAR_T)
    uub[1] = dyn.scale_state(uub[1], SCALE_MEAN_T, SCALE_VAR_T)
    if morphing_disabled is not None:
        ulb[2] = morphing_disabled-0.01
        uub[2] = morphing_disabled+0.01
    # if not (morphing_disabled is None):
    #     ulb[-1] = morphing_disabled
    #     uub[-1] = morphing_disabled
    lb = np.zeros(PARA_DIMENSION_VAR)
    ub = np.zeros(PARA_DIMENSION_VAR)
    lb[:PARA_INDEXES_VAR[6]] = -np.inf
    ub[:PARA_INDEXES_VAR[6]] = np.inf
    lb[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]] = np.concatenate([ulb for i in range(PARA_N_LGL_ALL)])
    ub[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[8]] = np.concatenate([uub for i in range(PARA_N_LGL_ALL)])
    bounds = Bounds(lb=lb, ub=ub)
    optimal_solution = minimize(fun=fun_obj_scaled, x0=Xinitial_scaled, constraints=constraints, bounds=bounds, args=t_switch, method='SLSQP', callback=callback_PS, options={'maxiter': 2500, 'disp': True, 'iprint': 2})
    x_optimal, y_optimal, z_optimal, u_optimal = unzip_variable(optimal_solution.x)
    j_optimal = optimal_solution.fun

    return x_optimal, y_optimal, z_optimal, u_optimal, j_optimal


def function_objective_casadi(X, t_switch, xi_ref_a, xi_ref_c):
    # y_last_aggre = X[PARA_N_LGL_ALL * PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE * PARA_NY_AUXILIARY - 2]
    # y_last_cruise = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-1]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    # y_last = casadi.vertcat(y_last_aggre, -y_last_cruise)
    # gy = y_last - casadi.MX([PARA_EPI12 * t_switch, -PARA_EPI22 * (PARA_TF - t_switch)])
    # max_gy = casadi.mmax(gy)
    # cost = casadi.fmax(max_gy, -z_last[0] + 0.05*(X[(PARA_N_LGL_ALL-1) * PARA_NX_AUXILIARY + 4] - 300)**2 / 10000) # ori
    # cost = casadi.fmax(max_gy, -z_last[0] + 0.02*(X[(PARA_N_LGL_ALL-1) * PARA_NX_AUXILIARY + 4] - 300)**2 / 10000)
    # cost = casadi.fmax(max_gy, z_last[0])

    # smooth loss
    # u0a = X[PARA_U0A_INDEX]
    # u1a = X[PARA_U1A_INDEX]
    u2a = X[PARA_U2A_INDEX]
    # u0c = X[PARA_U0C_INDEX]
    # u1c = X[PARA_U1C_INDEX]
    u2c = X[PARA_U2C_INDEX]

    # cost = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1] + (smooth_loss(u2a) + smooth_loss(u2c))

    cost = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1] + (ref_xi_loss(u2a, xi_ref_a) + ref_xi_loss(u2c, xi_ref_c))

    # cost = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]

    # cost = z_last[0]

    return cost

def function_objective_casadi_fuel(X, t_switch):
    y_last_aggre = X[PARA_N_LGL_ALL * PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE * PARA_NY_AUXILIARY - 2]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    gy = y_last_aggre - PARA_EPI12 * t_switch
    cost = casadi.fmax(gy, z_last[0])
    return cost

def function_objective_casadi_manu(X, t_switch):
    y_last_cruise = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-1]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    gy = -y_last_cruise + PARA_EPI22 * (PARA_TF - t_switch)
    cost = casadi.fmax(gy, z_last[0])
    return cost

def function_objective_casadi_both(X, t_switch):
    y_last_aggre = X[PARA_N_LGL_ALL * PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE * PARA_NY_AUXILIARY - 2]
    y_last_cruise = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-1]
    z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    y_last = casadi.vertcat(y_last_aggre, -y_last_cruise)
    gy = y_last - casadi.MX([PARA_EPI12 * t_switch, -PARA_EPI22 * (PARA_TF - t_switch)])
    max_gy = casadi.mmax(gy)
    cost = casadi.fmax(max_gy, z_last[0])
    return cost

def function_objective_casadi_LGR(X, t_switch):
    # y_last_aggre = X[PARA_N_LGL_ALL * PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE * PARA_NY_AUXILIARY - 2]
    # y_last_cruise = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY)-1]
    # # z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-PARA_NZ_AUXILIARY : PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)]
    # z_last = X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
    # y_last = casadi.vertcat(y_last_aggre, -y_last_cruise)
    # gy = y_last - casadi.MX([PARA_EPI12 * t_switch, -PARA_EPI22 * (PARA_TF - t_switch)])
    # max_gy = casadi.mmax(gy)
    # # cost = casadi.fmax(max_gy, -z_last[0] + 0.05*(X[(PARA_N_LGL_ALL-1) * PARA_NX_AUXILIARY + 4] - 300)**2 / 10000) # ori
    # # cost = casadi.fmax(max_gy, z_last[0] + 0.02*(X[(PARA_N_LGL_ALL-1) * PARA_NX_AUXILIARY + 4] - 300)**2 / 10000)
    # # cost = casadi.fmax(max_gy, z_last[0])
    # cost = z_last[0]
    # return -z_last[0]
    # cost = casadi.fmin(max_gy+50, -z_last[0])
    return X[PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)-1]
#
def function_constraint_casadi_LGR(X, t_switch, h_ref_lgl, diff_mat, x0):
    # t_switch = arg[0]
    # h_ref_lgl = arg[1]
    # diff_mat = arg[2]

    eq_cons_array = casadi.MX.zeros(DIM_CONS_LGR)

    x_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[1]], (PARA_NX_AUXILIARY, PARA_N_LGL_AGGRE)))  # Saved as row vector
    x_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[1]:PARA_INDEXES_VAR[2]], (PARA_NX_AUXILIARY, PARA_N_LGL_CRUISE)))
    y_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[2]:PARA_INDEXES_VAR[3]], (PARA_NY_AUXILIARY, PARA_N_LGL_AGGRE)))
    y_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[3]:PARA_INDEXES_VAR[4]], (PARA_NY_AUXILIARY, PARA_N_LGL_CRUISE)))
    z_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[4]:PARA_INDEXES_VAR[5]], (PARA_NZ_AUXILIARY, PARA_N_LGL_AGGRE)))
    z_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[5]:PARA_INDEXES_VAR[6]], (PARA_NZ_AUXILIARY, PARA_N_LGL_CRUISE)))
    u_aggre_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[6]:PARA_INDEXES_VAR[7]], (PARA_NU_AUXILIARY, PARA_N_LGL_AGGRE)))
    u_cruise_matrix = casadi.transpose(casadi.reshape(X[PARA_INDEXES_VAR[7]:PARA_INDEXES_VAR[8]], (PARA_NU_AUXILIARY, PARA_N_LGL_CRUISE)))

    # constraints for x
    fx_matrix_a = casadi.MX.zeros((PARA_N_LGL_AGGRE-1, PARA_NX_AUXILIARY))
    for m in range(PARA_N_LGL_AGGRE-1):
        fx_matrix_a[m, :] = dyn.dynamic_function_casadi(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    fx_matrix_a *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_LGR[0]:PARA_INDEXES_CONS_LGR[1]] = casadi.reshape(diff_mat @ x_aggre_matrix - fx_matrix_a, ((PARA_N_LGL_AGGRE-1)*PARA_NX_AUXILIARY, 1))
    
    fx_matrix_c = casadi.MX.zeros((PARA_N_LGL_CRUISE-1, PARA_NX_AUXILIARY))
    for m in range(PARA_N_LGL_CRUISE-1):
        fx_matrix_c[m, :] = dyn.dynamic_function_casadi(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    fx_matrix_c *= (PARA_TF-t_switch)/2    
    eq_cons_array[PARA_INDEXES_CONS_LGR[1]:PARA_INDEXES_CONS_LGR[2]] = casadi.reshape(diff_mat @ x_cruise_matrix - fx_matrix_c, ((PARA_N_LGL_CRUISE-1)*PARA_NX_AUXILIARY, 1))
    
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

    fy_matrix_1 = casadi.MX.zeros((PARA_N_LGL_AGGRE-1, 1))
    for m in range(PARA_N_LGL_AGGRE-1):
        fy_matrix_1[m] = x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM
    fy_matrix_1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_LGR[2]:PARA_INDEXES_CONS_LGR[2]+PARA_N_LGL_AGGRE-1] = diff_mat @ y_aggre_matrix[:, 0] - fy_matrix_1
    eq_cons_array[PARA_INDEXES_CONS_LGR[2]:PARA_INDEXES_CONS_LGR[3]] = diff_mat @ y_aggre_matrix[:, 0] - fy_matrix_1
    # eq_cons_array[PARA_INDEXES_CONS_LGR[2]+PARA_N_LGL_AGGRE-1:PARA_INDEXES_CONS_LGR[3]] = y_aggre_matrix[:, 1]

    fy_matrix_2 = casadi.MX.zeros((PARA_N_LGL_CRUISE-1, 1))
    for m in range(PARA_N_LGL_CRUISE-1):
        L, D, M, _ = dyn.aerodynamic_forces(x_cruise_matrix[m, :], u_cruise_matrix[m, :])
        pa = casadi.vertcat(-D / PARA_m - PARA_g * casadi.sin(x_cruise_matrix[m, 1]), L / PARA_m - PARA_g * casadi.cos(x_cruise_matrix[m, 1]), M / PARA_Jy)
        pa_norm = casadi.norm_2(pa)
        pa_norm_norm = pa_norm / PARA_PA_NORM
        fy_matrix_2[m] = pa_norm_norm
        # fy_matrix_2[m] = x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM
    fy_matrix_2 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_CONS_LGR[3]:PARA_INDEXES_CONS_LGR[3]+PARA_N_LGL_AGGRE] = y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0]
    # eq_cons_array[PARA_INDEXES_CONS_LGR[3]+PARA_N_LGL_AGGRE:PARA_INDEXES_CONS_LGR[4]] = diff_mat @ y_cruise_matrix[:, 1] - fy_matrix_2
    eq_cons_array[PARA_INDEXES_CONS_LGR[3]:PARA_INDEXES_CONS_LGR[4]] = diff_mat @ y_cruise_matrix[:, 1] - fy_matrix_2
    # eq_cons_array[PARA_INDEXES_VAR[3]+PARA_N_LGL_AGGRE:PARA_INDEXES_VAR[4]] = y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0]


    # constraints for z
    fz_matrix_a = casadi.MX.zeros((PARA_N_LGL_AGGRE-1, PARA_NZ_AUXILIARY))
    for m in range(PARA_N_LGL_AGGRE-1):
        fz_matrix_a[m, 0] = dyn.cost_tracking_error(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    fz_matrix_a *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS_LGR[4]:PARA_INDEXES_CONS_LGR[5]] = casadi.reshape(diff_mat @ z_aggre_matrix - fz_matrix_a, ((PARA_N_LGL_AGGRE-1)*PARA_NZ_AUXILIARY, 1))
    
    fz_matrix_c = casadi.MX.zeros((PARA_N_LGL_CRUISE-1, PARA_NZ_AUXILIARY))
    for m in range(PARA_N_LGL_CRUISE-1):
        fz_matrix_c[m, 0] = dyn.cost_tracking_error(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    fz_matrix_c *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS_LGR[5]:PARA_INDEXES_CONS_LGR[6]] = casadi.reshape(diff_mat @ z_cruise_matrix - fz_matrix_c, ((PARA_N_LGL_CRUISE-1)*PARA_NZ_AUXILIARY, 1))

    # constraints for link
    # link_index = PARA_INDEXES_VAR[6]+(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY)
    eq_cons_array[PARA_INDEXES_CONS_LGR[6]:PARA_INDEXES_CONS_LGR[7]] = casadi.horzcat(x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :])
    # eq_cons_array[PARA_INDEXES_VAR[6]+PARA_NX_AUXILIARY:PARA_INDEXES_VAR[6]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY] = 0
    # eq_cons_array[PARA_INDEXES_VAR[6]:link_index] = casadi.horzcat(x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], casadi.MX.zeros(1,2),
    #                                                         z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :])


    # constraints for start value
    eq_cons_array[PARA_INDEXES_CONS_LGR[7]:PARA_INDEXES_CONS_LGR[7]+PARA_NX_AUXILIARY] = x_aggre_matrix[0,:] - casadi.reshape(x0, (1, PARA_NX_AUXILIARY))
    eq_cons_array[PARA_INDEXES_CONS_LGR[7]+PARA_NX_AUXILIARY] = y_aggre_matrix[0, 0]
    eq_cons_array[PARA_INDEXES_CONS_LGR[7]+PARA_NX_AUXILIARY+1] = y_cruise_matrix[0, 1]
    eq_cons_array[PARA_INDEXES_CONS_LGR[7]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY: PARA_INDEXES_CONS_LGR[8]] = z_aggre_matrix[0,:]
    # eq_cons_array[link_index+PARA_NX_AUXILIARY:link_index+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY] = y_aggre_matrix[0, :]

    # constraints for control input
    for m in range(PARA_N_LGL_AGGRE):
        eq_cons_array[PARA_INDEXES_CONS_LGR[8] + m*PARA_NU_AUXILIARY: PARA_INDEXES_CONS_LGR[8] + (m+1)*PARA_NU_AUXILIARY] = u_aggre_matrix[m, :]
    start_index2 = PARA_INDEXES_CONS_LGR[8] + PARA_N_LGL_AGGRE*PARA_NU_AUXILIARY
    for m in range(PARA_N_LGL_CRUISE):
        eq_cons_array[start_index2 + m*PARA_NU_AUXILIARY: start_index2 + (m+1)*PARA_NU_AUXILIARY] = u_cruise_matrix[m, :]

    # constraints for velocity
    # vel_con_index = start_index2 + (m+1)*PARA_NU_AUXILIARY
    # for m in range(PARA_N_LGL_AGGRE):
    #     eq_cons_array[vel_con_index : vel_con_index+PARA_N_LGL_AGGRE] = x_aggre_matrix[m, 0]
    # for m in range(PARA_N_LGL_CRUISE):
    #     eq_cons_array[vel_con_index : vel_con_index+PARA_N_LGL_AGGRE] = x_cruise_matrix[m, 0]


    # return casadi.vertcat(eq_cons_array[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[2]], eq_cons_array[PARA_INDEXES_VAR[4]:])
    return eq_cons_array
    # return eq_cons_array


def function_constraint_casadi(X, t_switch, h_ref_lgl, diff_mat, x0, u0):
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
    fx_matrix =  casadi.MX.zeros(x_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fx_matrix[m, :] = dyn.dynamic_function_casadi(x=x_aggre_matrix[m, :], u=u_aggre_matrix[m, :])
    fx_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS[0]:PARA_INDEXES_CONS[1]] = casadi.reshape(diff_mat @ x_aggre_matrix - fx_matrix, (PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY, 1))
    
    fx_matrix = casadi.MX.zeros(x_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fx_matrix[m, :] = dyn.dynamic_function_casadi(x=x_cruise_matrix[m, :], u=u_cruise_matrix[m, :])
    fx_matrix *= (PARA_TF-t_switch)/2    
    eq_cons_array[PARA_INDEXES_CONS[1]:PARA_INDEXES_CONS[2]] = casadi.reshape(diff_mat @ x_cruise_matrix - fx_matrix, (PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY, 1))
    
    # constraints for y
    # V: x[0] T: u[1]
    fy_matrix_1 = casadi.MX.zeros((PARA_N_LGL_AGGRE, 1))
    for m in range(PARA_N_LGL_AGGRE):
        fy_matrix_1[m] = x_aggre_matrix[m, 0] * u_aggre_matrix[m, 1] / PARA_PC_NORM
    fy_matrix_1 *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS[2]:PARA_INDEXES_CONS[2]+PARA_N_LGL_AGGRE] = diff_mat @ y_aggre_matrix[:, 0] - fy_matrix_1
    # eq_cons_array[PARA_INDEXES_CONS[2]+PARA_N_LGL_AGGRE:PARA_INDEXES_CONS[3]] = y_aggre_matrix[:, 1]

    fy_matrix_2 = casadi.MX.zeros((PARA_N_LGL_CRUISE, 1))
    for m in range(PARA_N_LGL_CRUISE):
        L, D, M, _ = dyn.aerodynamic_forces(x_cruise_matrix[m, :], u_cruise_matrix[m, :])
        pa = casadi.vertcat(-D / PARA_m - PARA_g * casadi.sin(x_cruise_matrix[m, 1]), L / PARA_m - PARA_g * casadi.cos(x_cruise_matrix[m, 1]), M / PARA_Jy)
        pa_norm = casadi.norm_2(pa)
        pa_norm_norm = pa_norm / PARA_PA_NORM
        fy_matrix_2[m] = pa_norm_norm
        # fy_matrix_2[m] = x_cruise_matrix[m, 0] * u_cruise_matrix[m, 1] / PARA_PC_NORM
    fy_matrix_2 *= t_switch/2
    # eq_cons_array[PARA_INDEXES_CONS[3]:PARA_INDEXES_CONS[3]+PARA_N_LGL_AGGRE] = y_cruise_matrix[:, 0] - y_aggre_matrix[-1, 0]
    eq_cons_array[PARA_INDEXES_CONS[3]+PARA_N_LGL_AGGRE:PARA_INDEXES_CONS[4]] = diff_mat @ y_cruise_matrix[:, 1] - fy_matrix_2


    # constraints for z
    fz_matrix = casadi.MX.zeros(z_aggre_matrix.shape)
    for m in range(PARA_N_LGL_AGGRE):
        fz_matrix[m, 0] = dyn.cost_tracking_error(h=x_aggre_matrix[m, 4], h_r=h_ref_lgl[m])
    fz_matrix *= t_switch/2
    eq_cons_array[PARA_INDEXES_CONS[4]:PARA_INDEXES_CONS[5]] = casadi.reshape(diff_mat @ z_aggre_matrix - fz_matrix, (PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY, 1))
    
    fz_matrix = casadi.MX.zeros(z_cruise_matrix.shape)
    for m in range(PARA_N_LGL_CRUISE):
        fz_matrix[m, 0] = dyn.cost_tracking_error(h=x_cruise_matrix[m, 4], h_r=h_ref_lgl[m+PARA_N_LGL_AGGRE])
    fz_matrix *= (PARA_TF-t_switch)/2
    eq_cons_array[PARA_INDEXES_CONS[5]:PARA_INDEXES_CONS[6]] = casadi.reshape(diff_mat @ z_cruise_matrix - fz_matrix, (PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY, 1))

    # constraints for link
    eq_cons_array[PARA_INDEXES_CONS[6]:PARA_INDEXES_CONS[7]] = casadi.horzcat(x_aggre_matrix[-1,:]-x_cruise_matrix[0, :], z_aggre_matrix[-1,:]-z_cruise_matrix[0, :], u_aggre_matrix[-1,:]-u_cruise_matrix[0, :])

    # constraints for start value
    eq_cons_array[PARA_INDEXES_CONS[7]:PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY] = x_aggre_matrix[0,:] - casadi.reshape(x0, (1, PARA_NX_AUXILIARY))
    eq_cons_array[PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY] = y_aggre_matrix[0, 0]
    eq_cons_array[PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY+1] = y_cruise_matrix[0, 1]
    eq_cons_array[PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY: PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY] = z_aggre_matrix[0,:]
    eq_cons_array[PARA_INDEXES_CONS[7]+PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY:PARA_INDEXES_CONS[8]] = u_aggre_matrix[0,:] - casadi.reshape(u0, (1, PARA_NU_AUXILIARY))

    # constraints for control input
    for m in range(PARA_N_LGL_AGGRE):
        eq_cons_array[PARA_INDEXES_CONS[8] + m*PARA_NU_AUXILIARY: PARA_INDEXES_CONS[8] + (m+1)*PARA_NU_AUXILIARY] = u_aggre_matrix[m, :]
    start_index2 = PARA_INDEXES_CONS[8] + PARA_N_LGL_AGGRE*PARA_NU_AUXILIARY
    for m in range(PARA_N_LGL_CRUISE):
        eq_cons_array[start_index2 + m*PARA_NU_AUXILIARY: start_index2 + (m+1)*PARA_NU_AUXILIARY] = u_cruise_matrix[m, :]

    # constraints for velocity
    # vel_con_index = start_index2 + (m+1)*PARA_NU_AUXILIARY
    # for m in range(PARA_N_LGL_AGGRE):
    #     eq_cons_array[vel_con_index : vel_con_index+PARA_N_LGL_AGGRE] = x_aggre_matrix[m, 0]
    # for m in range(PARA_N_LGL_CRUISE):
    #     eq_cons_array[vel_con_index : vel_con_index+PARA_N_LGL_AGGRE] = x_cruise_matrix[m, 0]


    # return casadi.vertcat(eq_cons_array[PARA_INDEXES_VAR[0]:PARA_INDEXES_VAR[2]], eq_cons_array[PARA_INDEXES_VAR[4]:])
    return eq_cons_array
    # return eq_cons_array


def generate_PS_solution_casadi(x0, trajectory_ref, morphing_disabled=None, fun_obj = function_objective_casadi):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGL_points = calculate_LGL_points(N_LGL=PARA_N_LGL_AGGRE)  # Assume same number
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points, t_switch=t_switch)
    h_ref_lgl = h_ref[LGL_indexes]
    diff_mat = calculate_differential_matrix_LGL(PARA_N_LGL_AGGRE)  # Assume same number
    X0 = generate_initial_variables(x0=x0, traject_ref=trajectory_ref, method="pid")
    xi_ref_a = X0[PARA_U2A_INDEX]
    xi_ref_c = X0[PARA_U2C_INDEX]

    V = casadi.MX.sym("V", PARA_INDEXES_VAR[-1])
    J = fun_obj(X=V, t_switch=t_switch, xi_ref_a=xi_ref_a, xi_ref_c=xi_ref_c)
    g = function_constraint_casadi(X=V, t_switch=t_switch, h_ref_lgl=h_ref_lgl, diff_mat=diff_mat, x0=x0, u0=PARA_U0)

    dim_constraints = DIM_CONS
    lbg = np.zeros(dim_constraints)
    ubg = np.zeros(dim_constraints)
    # For feasible
    delta_n = (PARA_N_LGL_AGGRE)**(3/2-PARA_NX_AUXILIARY)
    # lbg[0: PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)] = -delta_n
    # ubg[0: PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)] = delta_n
    ulb = PARA_U_LOWER_BOUND.copy()
    uub = PARA_U_UPPER_BOUND.copy()
    if not (morphing_disabled is None):
        ulb[-1] = morphing_disabled
        uub[-1] = morphing_disabled
    lbg[PARA_INDEXES_CONS[8]: PARA_INDEXES_CONS[9]] = np.concatenate([ulb for i in range(PARA_N_LGL_ALL)], 0)
    ubg[PARA_INDEXES_CONS[8]: PARA_INDEXES_CONS[9]] = np.concatenate([uub for i in range(PARA_N_LGL_ALL)], 0)

    nlp = {'x':V, 'f':J, 'g':g}
    opts = {}
    opts["expand"] = True

    # opts["ipopt.max_iter"] = 100
    opts["ipopt.acceptable_tol"] = 1e-6
    # opts["ipopt.linear_solver"] = 'ma27'
    # opts["verbose"] = False
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


def generate_PS_solution_casadi_LGR(x0, trajectory_ref, morphing_disabled=None):
    t_switch = trajectory_ref['t_switch']
    h_ref = trajectory_ref['h_r_seq']
    LGR_points = calculate_LGR_points(N_LGR=PARA_N_LGL_AGGRE)  # Assume same number
    LGR_indexes, _ = calculate_LGR_indexes(LGR_points=LGR_points, t_switch=t_switch)
    h_ref_lgr = h_ref[LGR_indexes]
    diff_mat = calculate_differential_matrix_LGR(PARA_N_LGL_AGGRE)  # Assume same number
    X0 = generate_initial_variables_LGR(x0=x0, traject_ref=trajectory_ref, method="pid")

    V = casadi.MX.sym("V", PARA_INDEXES_VAR[-1])
    J = function_objective_casadi_LGR(X=V, t_switch=t_switch)
    g = function_constraint_casadi_LGR(X=V, t_switch=t_switch, h_ref_lgl=h_ref_lgr, diff_mat=diff_mat, x0=x0)

    dim_constraints = DIM_CONS_LGR
    lbg = np.zeros(dim_constraints)
    ubg = np.zeros(dim_constraints)
    # ueqcons_index = PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) \
    #        +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY) \
    #        +(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
    ulb = PARA_U_LOWER_BOUND.copy()
    uub = PARA_U_UPPER_BOUND.copy()
    if not (morphing_disabled is None):
        ulb[-1] = morphing_disabled
        uub[-1] = morphing_disabled
    lbg[PARA_INDEXES_CONS_LGR[8]: PARA_INDEXES_CONS_LGR[9]] = np.concatenate([ulb for i in range(PARA_N_LGL_ALL)], 0)
    ubg[PARA_INDEXES_CONS_LGR[8]: PARA_INDEXES_CONS_LGR[9]] = np.concatenate([uub for i in range(PARA_N_LGL_ALL)], 0)
    # ubg[ueqcons_index+PARA_N_LGL_ALL*PARA_NU_AUXILIARY: ueqcons_index+PARA_N_LGL_ALL*PARA_NU_AUXILIARY+PARA_N_LGL_ALL] = 100 * np.ones(PARA_N_LGL_ALL)

    nlp = {'x':V, 'f':J, 'g':g}
    opts = {}
    opts["expand"] = True

    # opts["ipopt.max_iter"] = 100
    opts["ipopt.acceptable_tol"] = 1e-5
    # opts["ipopt.linear_solver"] = 'ma27'
    # opts["verbose"] = False
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

    # for p in range(ajac.shape[0]): 
    #     for q in range(ajac.shape[1]): 
    #         print ([p, q, ajac[p, q]] if np.abs(ajac[p, q])>500 else "")

    # def fun(x, u):
    #     return dynamic_function(x, u)
