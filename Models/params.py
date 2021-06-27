import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols

W_acc_CW_s = sp.MatrixSymbol('acc_C', 3, 1)
R_WC_s = sp.MatrixSymbol('R_WC', 3, 3)
W_om_CW_s = sp.MatrixSymbol('om_C', 3, 1)
W_alp_CW_s = sp.MatrixSymbol('alp_C', 3, 1)

n_dofs = 10
q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]
dofs_s = [*q_s, *q_dot_s, *q_ddot_s]
n_dof_vector = len(dofs_s)

params_s = [W_acc_CW_s, R_WC_s, W_om_CW_s, W_alp_CW_s, *dofs_s]