from Filter import States, Filter, VisualTraj
from Models import SymProbe
import numpy as np
import sys

def parse_arguments():

    def print_usage():
        print(f"Usage: {__file__} <traj_name> <prop> <Rpval> <Rqval>")
        print("\t <traj_name>  - mandala0_mono, trans_x, rot_x, ...")
        print("\t <prop>       - prop / update")
        print("Optional:\n\t <Rpval>: default is 1e2")
        print("\t <Rqval>: default is 0.5")
        sys.exit()

    try:
        traj_name = sys.argv[1]
        do_prop_only = False or (sys.argv[2] == 'prop')

        print('Chosen settings: ')
        print(f'\t* Trajectory     : {traj_name}')
        print(f'\t* Propagate only : {do_prop_only}')
    except IndexError:
        print_usage()

    # optional args
    optional_args = len(sys.argv) > 3

    return traj_name, do_prop_only, optional_args
traj_name, do_prop_only, optional_args = parse_arguments()

# load data
from generate_data import get_data
from generate_data import probe, cov0
cam, cam_interp, imu, imu_ref, IC, min_t, max_t = get_data(traj_name)

# measurement noise values
Rpval = float(sys.argv[3]) if optional_args else 1e2
Rqval = float(sys.argv[4]) if optional_args else 0.5
meas_noise = np.hstack(([Rpval]*3, [Rqval]*4))
print(f'\n\tRp = {Rpval}\n\tRq = {Rqval}')
print(f'----------------------------')

# initialisation (t=0): IC, IMU buffer, noise matrices
sym_probe = SymProbe(probe)
imu.eval_init()
kf = Filter(imu, sym_probe, IC, cov0, meas_noise)
kf.traj.append_state(cam.t[0], kf.states)
# gain_plt = MatrixPlotter('K', min_row=0, min_col=0, max_row=3, max_col=3)

# filter main loop (t>=1)
old_t = min_t
cap_t = None
for i, t in enumerate(cam.t[1:]):

    # propagate
    # queue = imu.traj.get_queue(old_t, t) # real imu data
    # simulate imu queue
    queue = cam_interp.generate_queue(old_t, t)
    old_ti = old_t

    print(f"Predicting... t={queue.t[0]}")
    for ii, ti in enumerate(queue.t):
        interp = queue.at_index(ii)
        om, acc = imu.eval_expr_single(ti, *probe.joint_dofs,
            interp.acc, interp.R,
            interp.om, interp.alp, )
        imu_ref.append_value(ti, interp.vec)

        kf.dt = ti - old_ti
        kf.propagate(ti, om, acc)

        old_ti = ti

    # update
    if not do_prop_only:
        current_vis = cam.traj.at_index(i + 1) # not counting IC
        K = kf.update(current_vis)

        # error in calculating K -- stop sim
        if K is None:
            break

        # gain_plt.append(t, K)

    old_t = t

    if cap_t is not None and t >= cap_t:
        max_t = cap_t
        break

# plots
from plotter import plot_trajectories
plot_trajectories(kf.traj, traj_name, do_prop_only, imu, imu_ref, Rpval, Rqval)