import numpy as np
from spatialmath import SE3, SO3

x_00 = np.array([0, 0, 0]).reshape((3, 1))
F_00 = SE3()
R_act_10 = SO3.Rz(-45, unit="deg")
t = np.array([0, 1, 0]).reshape((3, 1))

T_t = SE3(t)
T_R = SE3(R_act_10)

# frame transformations (global)
F_01 = T_t * F_00
F_02 = T_R * T_t * F_00  # global/extrinsic
F_12 = F_01.inv() * F_02

# active transforms (global + local)
x_01 = T_t * x_00
x_02 = T_R * x_01
x_loc = T_t * T_R * x_00

# passive transforms (global + local)
x_10 = T_t.inv() * x_00
x_20 = T_t.inv() * T_R.inv() * x_00
x_loc_20 = T_R.inv() * T_t.inv() * x_00
