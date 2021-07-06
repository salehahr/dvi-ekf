import casadi

dt = casadi.SX.sym('dt')

# states
p_B = casadi.SX.sym('p_B', 3)
v_B = casadi.SX.sym('v_B', 3)
R_WB = casadi.SX.sym('R_WB', 3, 3)
dofs = casadi.SX.sym('q', 6)
p_C = casadi.SX.sym('p_C', 3)

x = [p_B, v_B, R_WB, dofs, p_C]
x_str = ['p_B', 'v_B', 'R_WB', 'dofs', 'p_C']

# inputs
acc = casadi.SX.sym('acc', 3)
om = casadi.SX.sym('om', 3)

u = [om, acc]
u_str = ['om', 'acc']

# error states
err_p_B = casadi.SX.sym('err_p_B', 3)
err_v_B = casadi.SX.sym('err_v_B', 3)
err_theta = casadi.SX.sym('err_theta', 3)
err_dofs = casadi.SX.sym('err_dofs', 6)
err_p_C = casadi.SX.sym('err_p_C', 3)

err_x = [err_p_B, err_v_B, err_theta, err_dofs, err_p_C]
err_x_str = ['err_p_B', 'err_v_B', 'err_theta', 'err_dofs', 'err_p_C']
            
# noise
n_v = casadi.SX.sym('n_v', 3)
n_om = casadi.SX.sym('n_om', 3)
n_dofs = casadi.SX.sym('n_dofs', 6)

n = [n_v, n_om, n_dofs]
n_str = ['n_v', 'n_om', 'n_dofs']

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

    # deriving err_p_C_dot -- define the true values
    v_B_tr = v_B + err_v_B
    R_WB_tr = R_WB @ (casadi.DM.eye(3) + casadi.skew(err_theta))
    dofs_tr = dofs + err_dofs
    om_tr = om - n_om

    # deriving err_p_C_dot -- continuous time
    p_CB_dot = R_WB @ probe.v \
            + casadi.skew(om) @ R_WB @ probe.p
    p_CB_dot_tr = R_WB_tr @ probe.v \
            + casadi.skew(om_tr) @ R_WB_tr @ probe.p

    p_C_dot = v_B + p_CB_dot
    p_C_dot_tr = v_B_tr + p_CB_dot_tr

    # err_p_C_dot = p_C_dot_tr - p_C_dot # results in free variables v_B
    err_p_C_dot = err_v_B + p_CB_dot_tr - p_CB_dot

    return err_p_C_dot
