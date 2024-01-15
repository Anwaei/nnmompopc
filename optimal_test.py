import numpy as np
import casadi
from matplotlib import pyplot as plt
from scipy.special import legendre

PARA_TF = 20
PARA_DT = 0.0001
PARA_STEP_NUM = int(PARA_TF/PARA_DT)+1
TIME_STEPS = np.arange(PARA_STEP_NUM) * PARA_DT
X0 = np.array([0, 10*2*np.pi/(PARA_TF/2)])
X0 = np.array([0, 0])
U0 = np.array([0])

NX = 2
NU = 1
NJ = 1
NZ = 1

PARA_KP = 2
PARA_KI = 0
PARA_KD = 1
PARA_KB = 0

N_LGL = 20

def control_origin_PID(ep, ei, ed):
    u = PARA_KB + PARA_KP * ep + PARA_KI * ei + PARA_KD * ed
    # uc = np.clip(u, 0, 1.5)
    return u

def control_origin_constant():
    u_const = 3
    return u_const

def dynamic_function(x, u):
    h = x[0]
    v = x[1]
    dx = np.array([v, u[0]])
    return dx

def dynamic_function_casadi(x, u):
    h = x[0]
    v = x[1]
    dx = casadi.horzcat(v, u[0])
    return dx

def dynamic_one_step(x, u):
    dt = PARA_DT
    dx = dynamic_function(x, u)
    new_x = x + dt*dx
    return new_x

def cost_func(x, u, h_r):
    err = (x[0] - h_r)**2
    return err

def generate_ref(type='sin'):
    A = 10
    w = 2*np.pi/(PARA_TF/2)
    h_r = A*np.sin(w*TIME_STEPS)
    return h_r

def simulate_origin(x0, trajectory_ref, control_method='pid', given_input=None):
    nx = NX
    nu = NU
    nz = NZ
    nj = NJ
    h_r_seq = trajectory_ref

    dt = PARA_DT 
    step_all = PARA_STEP_NUM

    u0 = U0

    x_all = np.zeros((step_all, nx))
    u_all = np.zeros((step_all, nu))
    z_all = np.zeros((step_all, nz))
    j_all = np.zeros((step_all, nj))
    f_all = np.zeros((step_all, 4))

    x_all[0, :] = x0
    if given_input is None:
        u_all[0, :] = u0
    else:
        u_all[0, :] = given_input[0, :]
    z_all[0, :] = 0
    j_all[0, :] = cost_func(x=x0, u=u0, h_r=h_r_seq[0])

    # For pid
    err_int = 0
    err_pre = 0

    for k in range(1, step_all):
        x_all[k, :] = dynamic_one_step(x=x_all[k-1, :], u=u_all[k-1, :])

        if control_method == "pid":
            err_cur = h_r_seq[k] - x_all[k, 0]
            err_int += err_cur
            err_diff = (err_cur - err_pre)/dt
            err_pre = err_cur
            u_all[k, :] = control_origin_PID(err_cur, err_int, err_diff)
        elif control_method == "given":
            u_all[k, :] = given_input[k, :]
        else:
            u_all[k, :] = control_origin_constant()

        j_all[k, :] = cost_func(x=x_all[k, :], u=u_all[k, :], h_r=h_r_seq[k])
        z_all[k, :] = z_all[k-1] - cost_func(x=x_all[k, :], u=u_all[k, :], h_r=h_r_seq[k]) * PARA_DT
    
    return x_all, z_all, u_all, j_all

def calculate_LGL_points(N_LGL):
    L = legendre(N_LGL-1)
    Ld = np.poly1d([i * L[i] for i in range(N_LGL-1, 0, -1)])
    LGL_points = np.append(Ld.r, [-1, 1])
    LGL_points.sort()    
    return LGL_points

def calculate_LGL_indexes(LGL_points):
    LGL_time = PARA_TF/2*LGL_points + PARA_TF/2
    LGL_indexes_float = LGL_time/PARA_DT
    LGL_indexes = LGL_indexes_float.astype(int).tolist()    
    return LGL_indexes, LGL_time

def calculate_LGL_weights(LGL_points, tau, j):
    weight = 1
    for i in range(N_LGL):
        if i != j:
            weight *= (tau-LGL_points[i])/(LGL_points[j]-LGL_points[i])
    return weight

def calculate_differential_matrix(N_LGL):
    LGL_points = calculate_LGL_points(N_LGL)
    D = np.zeros(shape=(N_LGL, N_LGL))
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


def function_objective_casadi(X):
    return -X[N_LGL * (NX + NZ)-NZ]

def function_constraint_casadi(X, h_r_seq, diff_mat):
    dim_constraint = N_LGL*(NX+NZ) + NX + NZ
    eq_cons_array = casadi.MX.zeros(dim_constraint)
    x_matrix = casadi.transpose(casadi.reshape(X[0: N_LGL*NX], (NX, N_LGL)))  # row vector
    z_matrix = casadi.transpose(casadi.reshape(X[N_LGL*NX: N_LGL*(NX+NZ)], (NZ, N_LGL)))
    u_matrix = casadi.transpose(casadi.reshape(X[N_LGL*(NX+NZ): N_LGL*(NX+NZ+NU)], (NU, N_LGL)))

    # constraints for x
    fx_matrix = casadi.MX.zeros(x_matrix.shape)
    for m in range(N_LGL):
        dx = dynamic_function_casadi(x=x_matrix[m, :], u=u_matrix[m, :])
        fx_matrix[m, :] = dx
    fx_matrix *= PARA_TF/2
    eq_cons_array[0: N_LGL*NX] = casadi.reshape(diff_mat@x_matrix - fx_matrix, (N_LGL*NX, 1))

    # constraints for z
    fz_matrix = casadi.MX.zeros(z_matrix.shape)
    for m in range(N_LGL):
        fz_matrix[m, :] = -cost_func(x=x_matrix[m, :], u=u_matrix[m, :], h_r=h_r_seq[m])
    fz_matrix *= PARA_TF/2
    eq_cons_array[N_LGL*NX: N_LGL*(NX+NZ)] = casadi.reshape(diff_mat@z_matrix - fz_matrix, (N_LGL*NZ, 1))

    # constraints for start value
    eq_cons_array[N_LGL*(NX+NZ): N_LGL*(NX+NZ)+NX] = x_matrix[0, :] - casadi.reshape(X0, (1, NX))
    eq_cons_array[N_LGL*(NX+NZ)+NX: N_LGL*(NX+NZ)+NX+NZ] = z_matrix[0, :]

    return eq_cons_array

def zip_variable(x_matrix, z_matrix, u_matrix):
    X = np.zeros(N_LGL * (NX + NZ + NU))
    X[0: N_LGL*NX] = np.reshape(x_matrix, N_LGL*NX)
    X[N_LGL*NX: N_LGL*(NX+NZ)] = np.reshape(z_matrix, N_LGL*NZ)
    X[N_LGL*(NX+NZ):N_LGL*(NX+NZ+NU)] = np.reshape(u_matrix, N_LGL*NU)
    return X

def interpolate_optimal_trajectory(x_optimal, z_optimal, u_optimal):
    LGL_points = calculate_LGL_points(N_LGL)
    t = TIME_STEPS
    x = np.zeros((PARA_STEP_NUM, NX))
    z = np.zeros((PARA_STEP_NUM, NZ))
    u = np.zeros((PARA_STEP_NUM, NU))
    for k in range(PARA_STEP_NUM):
        tau = 2/PARA_TF*t[k] - 1
        for j in range(N_LGL):
            w = calculate_LGL_weights(LGL_points, tau=tau, j=j)
            x[k, :] = x[k, :] + w * x_optimal[j, :]
            z[k, :] = z[k, :] + w * z_optimal[j, :]
            u[k, :] = u[k, :] + w * u_optimal[j, :]        
    return x, z, u

def unzip_variable(X):
    x_matrix = np.reshape(X[0: N_LGL*NX], (N_LGL, NX))  # Saved as row vector
    z_matrix = np.reshape(X[N_LGL*NX: N_LGL*(NX+NZ)], (N_LGL, NZ))
    u_matrix = np.reshape(X[N_LGL*(NX+NZ):N_LGL*(NX+NZ+NU)], (N_LGL, NU))
    return x_matrix, z_matrix, u_matrix


def plot_trajectory(x_all, z_all, u_all, j_all, trajectory_ref):
    plt.figure()
    plt.subplot(231)
    plt.plot(TIME_STEPS, x_all[:, 0])
    plt.plot(TIME_STEPS, trajectory_ref)
    plt.legend(["Actual h", "Target h"])
    plt.subplot(232)
    plt.plot(TIME_STEPS, x_all[:, 1])
    plt.legend(["Vel"])
    plt.subplot(233)
    plt.plot(TIME_STEPS, u_all[:, 0])
    plt.legend(["Input"])
    plt.subplot(234)
    plt.plot(TIME_STEPS, z_all[:, 0])
    plt.legend(["Z"])
    plt.subplot(235)
    plt.plot(TIME_STEPS, j_all)
    plt.legend(["Cost"])
    plt.show()

def plot_optimal_trajectory(v_opt, trajectory_ref, v_init):
    _, LGL_time = calculate_LGL_indexes(calculate_LGL_points(N_LGL))
    x_optimal, z_optimal, u_optimal = unzip_variable(v_opt)
    j_optimal = np.array(res["f"])
    x_all, z_all, u_all = interpolate_optimal_trajectory(x_optimal, z_optimal, u_optimal)
    j_all = np.zeros(PARA_STEP_NUM)
    for k in range(PARA_STEP_NUM):
        j_all[k] = cost_func(x=x_all[k, :], u=u_all[k, :], h_r=trajectory_ref[k])

    print(f"Final cost: {j_optimal}")

    x_init, z_init, u_init, j_init = v_init

    x_after, z_after, u_after, j_after = simulate_origin(X0, trajectory_ref, control_method='given', given_input=u_all)

    plt.figure()
    plt.subplot(231)
    plt.plot(TIME_STEPS, x_init[:, 0])
    plt.plot(TIME_STEPS, x_all[:, 0], color='darkorange')
    plt.plot(TIME_STEPS, x_after[:, 0], color='darkviolet')
    plt.plot(TIME_STEPS, trajectory_ref, color='lightgreen')
    plt.scatter(LGL_time, x_optimal[:, 0], color='darkorange')
    plt.legend(["Initial h", "Optimized h", "Controlled h", "Target h", "LGL points"])
    plt.subplot(232)
    plt.plot(TIME_STEPS, x_init[:, 1])
    plt.plot(TIME_STEPS, x_all[:, 1], color='darkorange')
    plt.plot(TIME_STEPS, x_after[:, 1], color='darkviolet')
    plt.scatter(LGL_time, x_optimal[:, 1], color='darkorange')
    plt.legend(["Initial v", "Optimized v", "Controlled v", "LGL points"])
    plt.subplot(233)
    plt.plot(TIME_STEPS, u_init[:, 0])
    plt.plot(TIME_STEPS, u_all[:, 0], color='darkorange')
    plt.plot(TIME_STEPS, u_after[:, 0], color='darkviolet')
    plt.scatter(LGL_time, u_optimal[:, 0], color='darkorange')
    plt.legend(["Initial u", "Optimized u", "Controlled u", "LGL points"])
    plt.subplot(234)
    plt.plot(TIME_STEPS, z_init[:, 0])
    plt.plot(TIME_STEPS, z_all[:, 0], color='darkorange')
    plt.plot(TIME_STEPS, z_after[:, 0], color='darkviolet')
    plt.scatter(LGL_time, z_optimal[:, 0], color='darkorange')
    plt.legend(["Initial z", "Optimized z", "Controlled z", "LGL points"])
    plt.subplot(235)
    plt.plot(TIME_STEPS, j_init)
    plt.plot(TIME_STEPS, j_all, color='darkorange')
    plt.plot(TIME_STEPS, j_after, color='darkviolet')
    plt.legend(["Initial j", "Optimized j", "Controlled j"])
    plt.show()

if __name__ == "__main__":
    h_r = generate_ref()
    x_init, z_init, u_init, j_init = simulate_origin(x0=X0, trajectory_ref=h_r)
    # plot_trajectory(x_init, z_init, u_init, j_init, h_r)
    LGL_points = calculate_LGL_points(N_LGL)
    diff_mat = calculate_differential_matrix(N_LGL)
    LGL_indexes, _ = calculate_LGL_indexes(LGL_points=LGL_points)
    h_r_lgl = h_r[LGL_indexes]
    Xinitial = zip_variable(x_init[LGL_indexes, :], z_init[LGL_indexes, :], u_init[LGL_indexes, :])

    # plt.scatter(LGL_points, np.zeros(N_LGL))
    plt.show()
    dim_variable = N_LGL * (NX + NZ + NU)
    V = casadi.MX.sym("V", dim_variable)
    J = function_objective_casadi(X=V)
    g = function_constraint_casadi(X=V, h_r_seq=h_r_lgl, diff_mat=diff_mat)
    dim_constraint = N_LGL*(NX+NZ) + NX + NZ
    lbg = np.zeros(dim_constraint)
    ubg = np.zeros(dim_constraint)

    nlp = {'x':V, 'f':J, 'g':g}
    opts = {}
    opts["expand"] = True
    #opts["ipopt.max_iter"] = 4
    # opts["ipopt.linear_solver"] = 'ma27'
    solver = casadi.nlpsol('S', 'ipopt', nlp, opts)
    res = solver(x0=Xinitial, lbg=lbg, ubg=ubg)
    print("optimal cost: ", float(res["f"]))
    v_opt = np.array(res["x"])
    x_optimal, z_optimal, u_optimal = unzip_variable(v_opt)
    j_optimal = np.array(res["f"])
    plot_optimal_trajectory(v_opt, h_r, [x_init, z_init, u_init, j_init])

