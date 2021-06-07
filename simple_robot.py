from Models import SimpleProbe

import sympy as sp

# initialise robot, joint variables
probe_BC = SimpleProbe(scope_length=0.5, theta_cam=sp.pi/6)

# set all joint values to 0
q_const = [q if not isinstance(q, sp.Expr) else 0. for q in probe_BC.q_sym]

# print and plot
if __name__ == '__main__':
    print(probe_BC)

    print(f"q_const: {q_const}")
    print(f"q_sym: {probe_BC.q_sym}")
    print(f"q_dot: {probe_BC.q_dot_sym}\n")

    probe_BC.plot(q_const)