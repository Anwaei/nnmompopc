import numpy as np

# ----- Environment Configurations -----
# Simulation time
PARA_TF = 200
PARA_DT = 0.01
PARA_STEP_NUM = int(PARA_TF/PARA_DT)+1
# State and control dimensions
PARA_NX_ORIGIN = 5
PARA_NU_ORIGIN = 3
PARA_NX_AUXILIARY = PARA_NX_ORIGIN
PARA_NU_AUXILIARY = PARA_NU_ORIGIN
PARA_NY_AUXILIARY = 2
PARA_NZ_AUXILIARY = 1
# Cost dimension
PARA_NJ_ORIGIN = 2
PARA_NJ_AUXILIARY = 1
# Parameters in the dynamics
PARA_m = 10
PARA_g = 9.8
PARA_Jy = 3.34
PARA_rho = 1.29
PARA_S = 0.84
PARA_cbar = 0.288
PARA_Sprop = 0.2027
PARA_Cprop = 1.0
PARA_Kmotor = 80
# Parameters in aerodynamic
PARA_C0 = {'L0': 0.19, 'Lalpha': 0.143, 'D0': 0.052, 'Dalpha': 0.00065, 'Dalpha2': 0.000325, 'M0': 0.195, 'Malpha': -0.057}
PARA_C1 = {'L0': -0.04, 'Lalpha': -0.032, 'D0': -0.0012, 'Dalpha': -0.000026, 'Dalpha2': -0.000013, 'M0': -0.065, 'Malpha': -0.143}
PARA_CLdeltae = -0.02
PARA_CMdeltae = 0.125
# Parameters of epsilon constraints
PARA_EPI12 = 10
PARA_EPI22 = 10
# Parameters of PS method
PARA_N_LGL_AGGRE = 15
PARA_N_LGL_CRUISE = 15
PARA_N_LGL_ALL = PARA_N_LGL_AGGRE + PARA_N_LGL_CRUISE
PARA_DIMENSION_VAR = PARA_N_LGL_ALL * (PARA_NX_AUXILIARY + PARA_NY_AUXILIARY + PARA_NZ_AUXILIARY + PARA_NU_AUXILIARY)
PARA_INDEXES_VAR = [0, 
    PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY, 
    PARA_N_LGL_ALL*PARA_NX_AUXILIARY, 
    PARA_N_LGL_ALL*PARA_NX_AUXILIARY + PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY, 
    PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY),
    PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY) + PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY, 
    PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY), 
    PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY) + PARA_N_LGL_AGGRE*PARA_NU_AUXILIARY,
    PARA_N_LGL_ALL*(PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY)]

# Cost normalization
PARA_PA_NORM = 20
PARA_PC_NORM = 3000
PARA_ERROR_SCALE = 10000

# Initial states
# Order: V, gamma, q, alpha, h
PARA_X0 = np.array([20, 0, 0, 3.359, 300])
# Order: delta_e, delta_T, xi
PARA_U0 = np.array([0.0, 0.5, 0.5])
# PARA_U0 = np.array([0.0, 0.0, 0.0])

# Parameters of control
# Order: delta_e, delta_T, xi
PARA_U_UPPER_BOUND = np.array([1.0, 1.0, 1.0])
PARA_U_LOWER_BOUND = np.array([0.0, 0.0, 0.0])
# Constant control
PARA_U_CONSTANT = np.array([0.1, 0.8, 0.5])
# PID control
PARA_KP = np.array([0.0, 0.5, 0.000])
PARA_KI = np.array([0.00, 1.2, 0.0])
PARA_KD = np.array([0.000, 0.000, 0.0])
PARA_KB = np.array([0.5, 0.5, 0.5])
# PARA_KP = np.array([0.000, 0.000, 0.000])
# PARA_KI = np.array([0.000, 0.000, 0.0])
