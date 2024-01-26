import numpy as np
import config_opc
import casadi


def aerodynamic_derivative(xi, key):
    return config_opc.PARA_C0[key] + config_opc.PARA_C1[key] * xi


def aerodynamic_coefficient_lift(alpha, xi, delta_e):
    return aerodynamic_derivative(xi, key='L0') + aerodynamic_derivative(xi, key='Lalpha') * alpha + config_opc.PARA_CLdeltae * delta_e


def aerodynamic_coefficient_drag(alpha, xi):
    return aerodynamic_derivative(xi, key='D0') + aerodynamic_derivative(xi, key='Dalpha') * alpha + aerodynamic_derivative(xi, key='Dalpha2') * (alpha**2)


def aerodynamic_coefficient_pitch_moment(alpha, xi, delta_e):
    return aerodynamic_derivative(xi, key='M0') + aerodynamic_derivative(xi, key='Malpha') * alpha + config_opc.PARA_CMdeltae * delta_e


def aerodynamic_forces(x, u):
    # V, gamma, q, alpha, h = x
    # delta_e, delta_T, xi = u
    V = x[0]
    alpha = x[1]
    q = x[2]
    theta = x[3]
    h = x[4]
    delta_e = u[0]
    delta_T = u[1]
    xi = u[2]

    aero_temp = 1 / 2 * config_opc.PARA_rho * (V**2) * config_opc.PARA_S
    L = aero_temp * aerodynamic_coefficient_lift(alpha, xi, delta_e)
    D = aero_temp * aerodynamic_coefficient_drag(alpha, xi)
    M = aero_temp * config_opc.PARA_cbar * aerodynamic_coefficient_pitch_moment(alpha, xi, delta_e)
    # T = 1 / 2 * config_opc.PARA_rho * config_opc.PARA_Sprop * config_opc.PARA_Cprop * (
    #             (config_opc.PARA_Kmotor * delta_T)**2 - V**2)
    T = delta_T

    return L, D, M, T


def dynamic_function(x, u):
    # V, gamma, q, alpha, h = x
    # delta_e, delta_T, xi = u
    V = x[0]
    gamma = x[1]
    q = x[2]
    alpha = x[3]
    h = x[4]
    delta_e = u[0]
    delta_T = u[1]
    xi = u[2]

    L, D, M, T = aerodynamic_forces(x, u)
    m = config_opc.PARA_m
    Jy = config_opc.PARA_Jy
    g = config_opc.PARA_g

    dx = np.array([1 / m * (T * np.cos(alpha) - D - m * g * np.sin(gamma)),
                   1 / (m * V) * (T * np.sin(alpha) + L - m * g * np.cos(gamma)),
                   M / Jy,
                   q - 1 / (m * V) * (T * np.sin(alpha) + L - m * g * np.cos(gamma)),
                   V * np.sin(gamma)])
    return dx


# def dynamic_function_casadi(x, u):
#     V = x[0]
#     gamma = x[1]
#     q = x[2]
#     alpha = x[3]
#     h = x[4]
#     delta_e = u[0]
#     delta_T = u[1]
#     xi = u[2]
#
#     L, D, M, T = aerodynamic_forces(x, u)
#     m = config_opc.PARA_m
#     Jy = config_opc.PARA_Jy
#     g = config_opc.PARA_g
#
#     dx = casadi.vertcat(1 / m * (T * casadi.cos(alpha) - D - m * g * casadi.sin(gamma)),
#                         1 / (m * V) * (T * casadi.sin(alpha) + L - m * g * casadi.cos(gamma)),
#                         M / Jy,
#                         q - 1 / (m * V) * (T * casadi.sin(alpha) + L - m * g * casadi.cos(gamma)),
#                         V * casadi.sin(gamma))
#     return dx

def dynamic_function_casadi(x, u):
    V = x[0]
    alpha = x[1]
    q = x[2]
    theta = x[3]
    h = x[4]
    delta_e = u[0]
    delta_T = u[1]
    xi = u[2]

    L, D, M, T = aerodynamic_forces(x, u)
    m = config_opc.PARA_m
    Jy = config_opc.PARA_Jy
    g = config_opc.PARA_g

    dx = casadi.vertcat(1 / m * (T * casadi.cos(alpha) - D - m * g * casadi.sin(theta-alpha)),
                        q - 1 / (m * V) * (T * casadi.sin(alpha) + L) + g * casadi.cos(theta-alpha)),
                        M / Jy,
                        q,
                        V * casadi.sin(theta-alpha))
    return dx


def dynamic_origin_one_step(x, u):
    dt = config_opc.PARA_DT
    dx = dynamic_function(x, u)
    new_x = x + dt * dx

    return new_x


def cost_tracking_error(h, h_r):
    return (h - h_r)**2 / config_opc.PARA_ERROR_SCALE


def cost_origin_cruise(x, u, h_r):  # h_r should be a scalar
    V, gamma, q, alpha, h = x
    delta_e, delta_T, xi = u

    T = 1 / 2 * config_opc.PARA_rho * config_opc.PARA_Sprop * config_opc.PARA_Cprop * (
                (config_opc.PARA_Kmotor * delta_T)**2 - V**2)
    P = np.abs(T) * V
    P_norm = P / config_opc.PARA_PC_NORM

    return np.array([cost_tracking_error(h, h_r), P_norm])


def cost_origin_cruise_casadi(x, u, h_r):  # h_r should be a scalar
    V, gamma, q, alpha, h = x[0], x[1], x[2], x[3], x[4]
    delta_e, delta_T, xi = u[0], u[1], u[2]

    T = 1 / 2 * config_opc.PARA_rho * config_opc.PARA_Sprop * config_opc.PARA_Cprop * (
                (config_opc.PARA_Kmotor * delta_T)**2 - V**2)
    P = casadi.fabs(T) * V
    P_norm = P / config_opc.PARA_PC_NORM

    return casadi.vertcat(cost_tracking_error(h, h_r), P_norm)


def cost_origin_aggressive(x, u, h_r):
    V, gamma, q, alpha, h = x[0], x[1], x[2], x[3], x[4]
    delta_e, delta_T, xi = u[0], u[1], u[2]

    L, D, M, _ = aerodynamic_forces(x, u)

    m = config_opc.PARA_m
    Jy = config_opc.PARA_Jy
    g = config_opc.PARA_g

    pa = np.array([-D / m - g * np.sin(alpha), L / m - g * np.cos(alpha), M / Jy])
    pa_norm = np.linalg.norm(pa)
    pa_norm_norm = pa_norm / config_opc.PARA_PA_NORM

    return np.array([cost_tracking_error(h, h_r), pa_norm_norm])


def cost_origin_aggressive_casadi(x, u, h_r):
    V, gamma, q, alpha, h = x[0], x[1], x[2], x[3], x[4]
    delta_e, delta_T, xi = u[0], u[1], u[2]

    L, D, M, _ = aerodynamic_forces(x, u)

    m = config_opc.PARA_m
    Jy = config_opc.PARA_Jy
    g = config_opc.PARA_g

    # pa = np.array([-D/m - g*casadi.sin(alpha), L/m - g * casadi.cos(alpha), M/Jy])
    # pa_norm = np.linalg.norm(pa)
    pa = casadi.vertcat(-D / m - g * casadi.sin(alpha), L / m - g * casadi.cos(alpha), M / Jy)
    pa_norm = casadi.norm_2(pa)
    pa_norm_norm = pa_norm / config_opc.PARA_PA_NORM

    return casadi.vertcat(cost_tracking_error(h, h_r), pa_norm_norm)
    # return np.array([cost_tracking_error(h, h_r), pa_norm_norm])


def cost_origin(x, u, h_r, t_current, t_switch):
    if t_current < t_switch:
        return cost_origin_aggressive(x, u, h_r)
    else:
        return cost_origin_cruise(x, u, h_r)


def dynamic_auxiliary_one_step(x, y, z, x_r, u, t, t_switch):
    new_x = dynamic_origin_one_step(x, u)
    # y: [y_(1,2), y_(2,2)]
    y1, y2 = y
    y1d = cost_origin_aggressive(x, u, x_r)[1] if t < t_switch else 0
    y2d = cost_origin_cruise(x, u, x_r)[1] if t >= t_switch else 0
    zd = -cost_tracking_error(x[-1], x_r)
    new_y = y + config_opc.PARA_DT * np.array([y1d, y2d])
    new_z = z + config_opc.PARA_DT * zd
    return new_x, new_y, new_z


def cost_auxiliary(y_f, z_f, t_switch):
    y12f = y_f[0]
    y22f = y_f[1]
    t1 = t_switch
    t2 = config_opc.PARA_TF - t_switch
    g12 = y12f + t1 * config_opc.PARA_EPI12
    g22 = y22f + t2 * config_opc.PARA_EPI22
    return np.max(np.array([-z_f[0], g12, g22]))


