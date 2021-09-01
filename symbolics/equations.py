import numpy as np
import casadi

from . import symbols as sym

###### EXPRESSIONS
""" IMU states = fcn(cam, probe)  - in W frame """
R_WB_exp    = sym.R_WC @ sym.R_BC.T
W_p_BW_exp  = sym.W_p_CW - R_WB_exp @ sym.B_p_CB

W_om_BW_exp = sym.W_om_CW - R_WB_exp @ sym.B_om_CB
W_omB_x_pCB = casadi.cross(W_om_BW_exp, R_WB_exp @ sym.B_p_CB)
WW_v_BW_exp = sym.WW_v_CW - R_WB_exp @ sym.BB_v_CB \
                - W_omB_x_pCB

W_alp_BW_exp = sym.W_alp_CW - R_WB_exp @ sym.B_alp_CB \
                - casadi.cross(W_om_BW_exp, R_WB_exp @ sym.B_om_CB)
W_acc_BW_exp = sym.W_acc_CW - R_WB_exp @ sym.B_acc_CB \
                - 2 * casadi.cross(W_om_BW_exp, R_WB_exp @ sym.BB_v_CB) \
                - casadi.cross(W_alp_BW_exp, R_WB_exp @ sym.B_p_CB) \
                - casadi.cross(W_om_BW_exp, W_omB_x_pCB)

""" IMU meas   = fcn(cam, probe)  - in B frame """
R_BW_exp    = sym.R_BC @ sym.R_WC.T
# B_om_BW_exp = R_BW_exp @ sym.W_om_CW - sym.B_om_CB
# B_omB_x_pCB = casadi.cross(B_om_BW_exp, self.B_p_CB)
# B_alp_BW_exp = R_BW_exp @ sym.W_alp_CW - sym.B_alp_CB \
                # - casadi.cross(B_om_BW_exp, sym.B_om_CB)
# B_acc_BW_exp = R_BW_exp @ sym.W_acc_CW \
                # - sym.B_acc_CB \
                # - 2 * casadi.cross(B_om_BW_exp, sym.BB_v_CB) \
                # - casadi.cross(B_alp_BW_exp, sym.B_p_CB) \
                # - casadi.cross(B_om_BW_exp, B_omB_x_pCB)

B_om_BW_exp  = R_BW_exp @ W_om_BW_exp
B_acc_BW_exp = R_BW_exp @ W_acc_BW_exp

""" Averaged IMU measurements """
B_om_BW_avg_exp = (sym.B_om_BW + sym.B_om_BW_n)/2

R_WB_curr_exp = sym.R_WB + sym.R_WB @ casadi.skew(sym.dt * B_om_BW_avg_exp)
R_acc_avg_exp = (sym.R_WB @ sym.B_acc_BW + R_WB_curr_exp @ sym.B_acc_BW_n)/2

""" Camera """
C_om_CW_exp = sym.R_BC.T @ (sym.B_om_BW + sym.B_om_CB)

###### FUNCTIONS
""" IMU reference trajectory (analytical) """
f_imu = casadi.Function('f_imu',
            [*sym.cam, *sym.probe_fwkin],
            [W_p_BW_exp, R_WB_exp, WW_v_BW_exp],
            [*sym.cam_str, *sym.probe_fwkin_str],
            ['W_p_BW', 'R_WB', 'WW_v_BW'])

""" IMU measurements u = func(cam, dofs) """
f_imu_meas = casadi.Function('f_imu_meas',
            [*sym.probe_fwkin],
            [B_om_BW_exp, B_acc_BW_exp],
            [*sym.probe_fwkin_str],
            ['B_om_BW', 'B_acc_BW'])

""" Kalman filter stuff """
f_predict = casadi.Function('f_nom',
    [sym.dt, *sym.x, *sym.u, *sym.probe_fwkin, *sym.u_current],
    [   sym.p_B + sym.dt * sym.v_B \
            + (sym.dt**2 / 2) * R_acc_avg_exp,
        sym.v_B + sym.dt * R_acc_avg_exp,
        R_WB_curr_exp,
        sym.dofs,
        sym.notch + sym.dt * sym.notchd,
        sym.notchd + sym.dt * sym.notchdd,
        sym.notchdd,
        sym.p_C \
            + sym.dt * sym.v_B \
            + sym.dt * sym.R_WB @ (sym.BB_v_CB + \
                casadi.cross(B_om_BW_avg_exp, sym.B_p_CB)),
        sym.R_WC_kf + sym.R_WC_kf @ casadi.skew(sym.dt * C_om_CW_exp)],
    ['dt', *sym.x_str, *sym.u_str, *sym.probe_fwkin_str, *sym.u_current_str],
    ['p_B_next', 'v_B_next', 'R_WB_next',
        'dofs_next', 'notch_next', 'notchd_next', 'notchdd_next', 
        'p_C_next', 'R_WC_next'])