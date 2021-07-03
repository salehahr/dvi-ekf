import numpy as np
import sympy as sp
import casadi

n_dofs = 8
n_dof_vector = n_dofs * 3

# sympy
W_acc_CW_s = sp.MatrixSymbol('acc_C', 3, 1)
R_WC_s = sp.MatrixSymbol('R_WC', 3, 3)
W_om_CW_s = sp.MatrixSymbol('om_C', 3, 1)
W_alp_CW_s = sp.MatrixSymbol('alp_C', 3, 1)

q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
qd_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
qdd_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]
dofs_s = [*q_s, *qd_s, *qdd_s]
params_s = [W_acc_CW_s, R_WC_s, W_om_CW_s, W_alp_CW_s, *dofs_s]

# casadi
W_acc_CW_cas = casadi.SX.sym('acc_C', 3)
R_WC_cas = casadi.SX.sym('R_WC', 3, 3)
W_om_CW_cas = casadi.SX.sym('om_C', 3)
W_alp_CW_cas = casadi.SX.sym('alp_C', 3)

q_cas = casadi.SX.sym('q', n_dofs)
qd_cas = casadi.SX.sym('qd', n_dofs)
qdd_cas = casadi.SX.sym('qdd', n_dofs)
dofs_cas = casadi.vertcat(q_cas, qd_cas, qdd_cas)