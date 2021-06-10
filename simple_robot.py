from Models import RigidSimpleProbe

import sympy as sp

# initialise robot, joint variables
probe_BtoC = RigidSimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

# print and plot
if __name__ == '__main__':
    print(probe_BtoC)

    print(f"q: {probe_BtoC.q}")
    print(f"q_sym: {probe_BtoC.q_sym}")
    print(f"q_dot: {probe_BtoC.q_dot_sym}\n")

    probe_BtoC.plot(probe_BtoC.q)