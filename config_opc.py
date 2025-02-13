import numpy as np

# ----- Environment Configurations -----
# Simulation time
PARA_TF = 100
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
PARA_m = 11.4
PARA_g = 9.8
PARA_Jy = 3.34
PARA_rho = 1.29
PARA_S = 0.84
PARA_cbar = 0.288
PARA_Sprop = 0.2027
PARA_Cprop = 1.0
PARA_Kmotor = 80
PARA_tau_xi = 1.5
PARA_K_xi = 1
# Parameters in aerodynamic
PARA_C0 = {'L0': 0.19, 'Lalpha': 0.143, 'D0': 0.052, 'Dalpha': 0.00065, 'Dalpha2': 0.000325, 'M0': 0.195, 'Malpha': -0.057}
PARA_C1 = {'L0': -0.04, 'Lalpha': -0.032, 'D0': -0.0012, 'Dalpha': -0.000026, 'Dalpha2': -0.000013, 'M0': -0.065, 'Malpha': -0.143}
# PARA_C0 = {'L0': 0.19, 'Lalpha': 0.143, 'D0': 0.052, 'Dalpha': 0.00065, 'Dalpha2': 0.000325, 'M0': 0.195, 'Malpha': -0.057}
# PARA_C1 = {'L0': -0.04, 'Lalpha': -0.032, 'D0': -0.0012, 'Dalpha': -0.000026, 'Dalpha2': -0.000013, 'M0': 0, 'Malpha': 0}
PARA_J0 = 3.04
PARA_J1 = 0.6
# PARA_J0 = 3.04
# PARA_J1 = 0
PARA_CLdeltae = -0.02
PARA_CMdeltae = 0.125
# Parameters of epsilon constraints
PARA_EPI12 = 0.04
# PARA_EPI12 = 0.03
PARA_EPI22 = 0.3
# Parameters of PS method
PARA_N_COLLECT = 35
# PARA_N_LGL_AGGRE =  35
# PARA_N_LGL_CRUISE = 35
PARA_N_LGL_AGGRE =  PARA_N_COLLECT
PARA_N_LGL_CRUISE = PARA_N_COLLECT
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
PARA_INDEXES_CONS = np.cumsum(
    [0,
     PARA_N_LGL_AGGRE*PARA_NX_AUXILIARY, 
     PARA_N_LGL_CRUISE*PARA_NX_AUXILIARY,
     PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY, 
     PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY,
     PARA_N_LGL_AGGRE*PARA_NZ_AUXILIARY, 
     PARA_N_LGL_CRUISE*PARA_NZ_AUXILIARY,
     (PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY), 
     (PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY), 
     PARA_N_LGL_ALL*PARA_NU_AUXILIARY
     ]
)
# PARA_INDEXES_CONS_LGR = np.cumsum(
#                      [0,
#                      (PARA_N_LGL_AGGRE-1)*PARA_NX_AUXILIARY,
#                      (PARA_N_LGL_CRUISE-1)*PARA_NX_AUXILIARY,
#                      PARA_N_LGL_AGGRE*PARA_NY_AUXILIARY-1,
#                      PARA_N_LGL_CRUISE*PARA_NY_AUXILIARY-1,
#                      (PARA_N_LGL_AGGRE-1)*PARA_NZ_AUXILIARY,
#                      (PARA_N_LGL_CRUISE-1)*PARA_NZ_AUXILIARY,
#                      (PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY),
#                      (PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY),
#                      PARA_N_LGL_ALL*PARA_NU_AUXILIARY
#                      ]).tolist()
PARA_INDEXES_CONS_LGR = np.cumsum(
                     [0,
                     (PARA_N_LGL_AGGRE-1)*PARA_NX_AUXILIARY,
                     (PARA_N_LGL_CRUISE-1)*PARA_NX_AUXILIARY,
                     (PARA_N_LGL_AGGRE-1)*(PARA_NY_AUXILIARY-1),
                     (PARA_N_LGL_CRUISE-1)*(PARA_NY_AUXILIARY-1),
                     (PARA_N_LGL_AGGRE-1)*PARA_NZ_AUXILIARY,
                     (PARA_N_LGL_CRUISE-1)*PARA_NZ_AUXILIARY,
                     (PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY),
                     (PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY),
                     PARA_N_LGL_ALL*PARA_NU_AUXILIARY
                     ]).tolist()
PARA_INDEXES_CONS_NP_LGL = np.cumsum(
                     [0,
                     (PARA_N_LGL_AGGRE)*PARA_NX_AUXILIARY,
                     (PARA_N_LGL_CRUISE)*PARA_NX_AUXILIARY,
                     (PARA_N_LGL_AGGRE)*PARA_NY_AUXILIARY,
                     (PARA_N_LGL_CRUISE)*PARA_NY_AUXILIARY,
                     (PARA_N_LGL_AGGRE)*PARA_NZ_AUXILIARY,
                     (PARA_N_LGL_CRUISE)*PARA_NZ_AUXILIARY,
                     (PARA_NX_AUXILIARY+PARA_NZ_AUXILIARY+PARA_NU_AUXILIARY),
                     (PARA_NX_AUXILIARY+PARA_NY_AUXILIARY+PARA_NZ_AUXILIARY)
                     ]).tolist()

# Cost normalization
PARA_PA_NORM = 10
PARA_PC_NORM = 10000
# PARA_ERROR_SCALE = 10000
PARA_ERROR_SCALE = 10000
# PARA_ERROR_SCALE = 1

# Initial states
# Order: V, gamma, q, alpha, h
# PARA_X0 = np.array([20, 0, 0, 3.359, 300])
# Order2: V, alpha, q, theta, h
PARA_X0 = np.array([20, 10/180*np.pi, -6/180*np.pi, 3/180*np.pi, 300])
# PARA_X0 = np.array([20, 10/180*np.pi, -6/180*np.pi, 3/180*np.pi, 300, 0.5])
# Order: delta_e, delta_T, xi
PARA_U0 = np.array([0.0, 30, 0.5])
# PARA_U0 = np.array([0.0, 0.0, 0.0])

# Parameters of control
# Order: delta_e, delta_T, xi
PARA_U_UPPER_BOUND = np.array([15/180*np.pi, 100, 0.9])
PARA_U_LOWER_BOUND = np.array([-15/180*np.pi, 15, 0.1])
# Constant control
PARA_U_CONSTANT = np.array([0.1, 0.8, 0.5])
PARA_FIX_XI = 0.5
# PID control
# PARA_KP = np.array([0.0002, 0.15, 0.001])
# PARA_KP = np.array([0.00012, 0.12, 0.0006])
# PARA_KP = np.array([0.0003, 0.2, 0.0015])
PARA_KP = np.array([0.0002, 0.2, 0.0020])
PARA_KI = np.array([0.00, 0.07, 0.001])
PARA_KD = np.array([0.000, 0.0, 0.0])
# PARA_KP = np.array([0.0003, 0.2, 0.0015])
# PARA_KI = np.array([0.0001, 0.02, 0.01])
# PARA_KD = np.array([0.000, 0.0, 0.0])

PARA_KP_L = np.array([0.0001, 0.1, 0.0005])
PARA_KP_U = np.array([0.0002, 0.2, 0.0020])

PARA_KB = np.array([0.5, 0.5, 0.5])




# ----- NN Model Configurations -----
AUX_STATES_DIM = 5+2+1+1
HIDDEN_DIM = 32
TIME_DIM = 1
REF_DIM = 10001
Q_DIM = 64
K_DIM = 64
V_DIM = 64

LEARNING_RATE = 0.006
MOMENTUM = 0.9
BATCH_SIZE = 20
TRAIN_PROP = 0.8
EPOCHES = 1000

#
DATA_PATH = 'data/opt_data_02-12-1252.pt'
STAT_PATH = 'data/opt_stats_02-12-1252.npz'
NET_PATH = 'model/net_08-26-2042/epoch_443.pth'

PARA_WP = 0.01
PARA_WN = 1
PARA_ELOSSSCALE = 1e5
PARA_XILOSSSCALE = 3e6
PARA_U0A_INDEX = np.arange(start=PARA_INDEXES_VAR[6]+0, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[6]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-2).tolist()
PARA_U1A_INDEX = np.arange(start=PARA_INDEXES_VAR[6]+1, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[6]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-1).tolist()
PARA_U2A_INDEX = np.arange(start=PARA_INDEXES_VAR[6]+2, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[6]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-0).tolist()
PARA_U0C_INDEX = np.arange(start=PARA_INDEXES_VAR[7]+0, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[7]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-2).tolist()
PARA_U1C_INDEX = np.arange(start=PARA_INDEXES_VAR[7]+1, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[7]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-1).tolist()
PARA_U2C_INDEX = np.arange(start=PARA_INDEXES_VAR[7]+2, step=PARA_NU_AUXILIARY, stop=PARA_INDEXES_VAR[7]+PARA_NU_AUXILIARY*PARA_N_LGL_AGGRE-0).tolist()


SCALE_MEAN_H = 0
SCALE_VAR_H = 50
SCALE_MEAN_V = 0
SCALE_VAR_V = 20
SCALE_MEAN_T = 50
SCALE_VAR_T = 50
