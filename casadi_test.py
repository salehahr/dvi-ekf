from casadi import *

p_B = SX.sym('p_B', 3)
v_B = SX.sym('v_B', 3)
R_WB = SX.sym('R_WB', 3, 3)
dt = SX.sym('dt')

p_B_kk = p_B + dt*v_B + dt**2/2*R_WB

p_C_kk = p_B_kk

print(p_C_kk)