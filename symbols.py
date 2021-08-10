import numpy as np
import sympy as sp
import casadi

### ROBOT KINEMATICS
num_dofs = 8
n_dof_vector = num_dofs * 3

B_p_CB = casadi.SX.sym('B_p_CB', 3)
R_BC = casadi.SX.sym('R_BC', 3, 3)

BB_v_CB = casadi.SX.sym('BB_v_CB', 3)
B_om_CB = casadi.SX.sym('B_om_CB', 3)

B_acc_CB = casadi.SX.sym('B_acc_CB', 3)
B_alp_CB = casadi.SX.sym('B_alp_CB', 3)

probe_fwkin = [B_p_CB, R_BC, BB_v_CB, B_om_CB, B_acc_CB, B_alp_CB]
probe_fwkin_str = ['B_p_CB', 'R_BC',
                    'BB_v_CB', 'B_om_CB',
                    'B_acc_CB', 'B_alp_CB']

# sympy params
q_s = [sp.Symbol(f'q{x}') for x in range(1,num_dofs+1)]
qd_s = [sp.Symbol(f'q{x}_dot') for x in range(1,num_dofs+1)]
qdd_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,num_dofs+1)]
dofs_s = [*q_s, *qd_s, *qdd_s]

# casadi params
W_acc_CW_cas = casadi.SX.sym('acc_C', 3)
R_WC_cas = casadi.SX.sym('R_WC', 3, 3)
W_om_CW_cas = casadi.SX.sym('om_C', 3)
W_alp_CW_cas = casadi.SX.sym('alp_C', 3)

q_cas = casadi.SX.sym('q', num_dofs)
qd_cas = casadi.SX.sym('qd', num_dofs)
qdd_cas = casadi.SX.sym('qdd', num_dofs)

err_q_cas = casadi.SX.sym('err_q', num_dofs)
q_tr_cas = q_cas + err_q_cas

dofs_cas = casadi.vertcat(q_cas, qd_cas, qdd_cas)
dofs_cas_list = casadi.vertsplit(dofs_cas)

### IMU
B_acc_BW = casadi.SX.sym('B_acc_BW', 3)
B_om_BW = casadi.SX.sym('B_om_BW', 3)

### CAMERA
W_p_CW = casadi.SX.sym('W_p_CW', 3)
R_WC = casadi.SX.sym('R_WC', 3, 3)

WW_v_CW = casadi.SX.sym('WW_v_CW', 3)
W_om_CW = casadi.SX.sym('W_om_CW', 3)

W_acc_CW = casadi.SX.sym('W_acc_CW', 3)
W_alp_CW = casadi.SX.sym('W_alp_CW', 3)

cam = [W_p_CW, R_WC, WW_v_CW, W_om_CW, W_acc_CW, W_alp_CW]
cam_str = ['W_p_CW', 'R_WC', 'WW_v_CW', 'W_om_CW', 'W_acc_CW', 'W_alp_CW']

C_om_CW = R_BC.T @ (B_om_BW + B_om_CB)

### KALMAN FILTER
dt = casadi.SX.sym('dt')

# states
p_B = casadi.SX.sym('p_B', 3)
v_B = casadi.SX.sym('v_B', 3)
R_WB = casadi.SX.sym('R_WB', 3, 3)
dofs, _ = casadi.vertsplit(q_cas, [0, 6, 8])

p_C = casadi.SX.sym('p_C', 3)
R_WC_kf = casadi.SX.sym('R_WC', 3, 3)

x = [p_B, v_B, R_WB, dofs, p_C, R_WC_kf]
x_str = ['p_B', 'v_B', 'R_WB', 'dofs', 'p_C', 'R_WC']

# inputs
u = [B_om_BW, B_acc_BW]
u_str = ['B_om_BW', 'B_acc_BW']

# error states
err_p_B = casadi.SX.sym('err_p_B', 3)
err_v_B = casadi.SX.sym('err_v_B', 3)
err_theta = casadi.SX.sym('err_theta', 3)
err_dofs, _ = casadi.vertsplit(err_q_cas, [0, 6, 8])
err_p_C = casadi.SX.sym('err_p_C', 3)
err_theta_C = casadi.SX.sym('err_theta_C', 3)

err_x = [err_p_B, err_v_B, err_theta, err_dofs, err_p_C, err_theta_C]
err_x_str = ['err_p_B', 'err_v_B', 'err_theta', 'err_dofs',
                'err_p_C', 'err_theta_C']
            
# noise
n_a = casadi.SX.sym('n_a', 3)
n_om = casadi.SX.sym('n_om', 3)
n_dofs = casadi.SX.sym('n_dofs', 6)

n = [n_a, n_om, n_dofs]
n_str = ['n_a', 'n_om', 'n_dofs']

# true values
# v_B_tr = v_B + err_v_B
R_WB_tr = R_WB @ (casadi.DM.eye(3) + casadi.skew(err_theta))
dofs_tr = dofs + err_dofs
om_tr = B_om_BW - n_om

def get_err_pc_dot(probe):
    """
        Example derivation of err_p_B:

        [In continuous time]
        p_B_tr_dot = p_B_dot + err_p_B_dot
        v_B_tr = v_B + err_v_B

        err_p_B_dot = v_B_tr - v_B
                    = v_B + err_v_B - v_B
                    = err_v_B

        [Discretised]
        err_p_B_next = err_p_B + dt * err_v_B
    """

    # deriving err_p_C_dot -- continuous time
    p_CB_dot = R_WB @ (probe.v + casadi.cross(B_om_BW, probe.p))
    p_CB_dot_tr = R_WB_tr @ (probe.v_tr + casadi.cross(om_tr, probe.p_tr))

    p_C_dot = v_B + p_CB_dot
    p_C_dot_tr = v_B + err_v_B + p_CB_dot_tr

    # err_p_C_dot = p_C_dot_tr - p_C_dot # results in free variables v_B
    err_p_C_dot = err_v_B + p_CB_dot_tr - p_CB_dot

    return err_p_C_dot

def rvec_to_quat(v):
    """ returns wxyz """
    return casadi.vertcat(0, v)

def quat_matrix(wxyz, dir):
    w, v = casadi.vertsplit(wxyz, [0, 1, 4])
    matr = casadi.SX.zeros(4, 4)
    matr[0,0] = w
    matr[0,1:] = -v
    matr[1:,0] = v

    if dir.lower() == 'r':
        matr[1:,1:] = w * casadi.SX.eye(3) - casadi.skew(v)
    elif dir.lower() == 'l':
        matr[1:,1:] = w * casadi.SX.eye(3) + casadi.skew(v)
    else:
        print("Invalid argument to quat_matrix.")
        raise Exception

    return matr

def get_err_theta_c_dot(probe):
    om_c = probe.R.T @ (B_om_BW + probe.om)
    om_c_tr = probe.R_tr.T @ (om_tr + probe.om_tr)

    om_c_quat = rvec_to_quat(om_c)
    om_c_tr_quat = rvec_to_quat(om_c_tr)

    err_q_C = casadi.vertcat(1, 0.5*err_theta_C)
    M_om = quat_matrix(om_c_tr_quat, 'r') - quat_matrix(om_c_quat, 'l')
    res = M_om @ err_q_C
    assert(res.shape == (4, 1))

    _, err_theta_c_dot = casadi.vertsplit(res, [0, 1, 4])

    return err_theta_c_dot