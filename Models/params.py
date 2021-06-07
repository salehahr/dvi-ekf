import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols

om_C_s = sp.MatrixSymbol('om_C', 3, 1)
acc_C_s = sp.MatrixSymbol('acc_C', 3, 1)
alp_C_s = sp.MatrixSymbol('alp_C', 3, 1)
R_BW_s = sp.MatrixSymbol('R_BW', 3, 3)

n_dofs = 10
q_s = [sp.Symbol(f'q{x}') for x in range(1,n_dofs+1)]
q_dot_s = [sp.Symbol(f'q{x}_dot') for x in range(1,n_dofs+1)]
q_ddot_s = [sp.Symbol(f'q{x}_ddot') for x in range(1,n_dofs+1)]
dofs_s = [*q_s, *q_dot_s, *q_ddot_s]
n_dof_vector = len(dofs_s)

params_s = [om_C_s, acc_C_s, alp_C_s, R_BW_s, *dofs_s]