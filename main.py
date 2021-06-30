from Filter import States, Filter, VisualTraj, MatrixPlotter
import numpy as np

def parse_arguments():
    import sys

    def print_usage():
        print(f"Usage: {__file__} <prop> <noise> [<vel>]")
        print("\t <prop>  - prop / all")
        print("\t <noise> - noise / nonoise")
        print("Optional arguments:")
        print("\t <vel>   - vel")
        sys.exit()

    try:
        do_prop_only = False or (sys.argv[1] == 'prop')
        use_noisy_imu = False or (sys.argv[2] == 'noise')
    except:
        print_usage()

    try:
        do_plot_vel =  (sys.argv[3] == 'vel')
    except:
        do_plot_vel = False

    return do_prop_only, use_noisy_imu, do_plot_vel
do_prop_only, use_noisy_imu, do_plot_vel = parse_arguments()

# load data
from generate_data import probe_BtoC, cam, imu
from generate_data import IC, cov0, min_t, max_t, gen_noise_matrices

# noise values
Qval = 1e-3
Rpval, Rqval = 1e3, 0.05

# initialisation (t=0)
kf = Filter(imu, IC, cov0, num_meas=7, num_control=6)
kf.init_imu(*probe_BtoC.joint_dofs)
kf.Qc, kf.R = gen_noise_matrices(Qval, Rpval, Rqval)

# filter main loop (t>=1)
old_t = min_t
for i, t in enumerate(cam.t[1:]):

    current_vis = cam.traj.at_index(i)

    # propagate
    imu_queue = imu.traj.get_queue(old_t, t)
    if imu_queue:
        old_ti = t
        for ii, ti in enumerate(imu_queue.t):
            current_imu = imu_queue.at_index(ii)
            kf.dt = ti - old_ti

            kf.propagate(ti, om, acc)

            old_ti = ti

    # update
    if not do_prop_only:
        kf.update(current_vis)

    old_t = t



# plots
import matplotlib.pyplot as plt
from plotter import plot_savefig, plot_trajectories, plot_velocities, plot_noise_sensitivity

plot_trajectories(kf.traj, do_prop_only)
# plot_velocities(kf.traj, do_plot_vel)
# plot_noise_sensitivity(kf.traj, Qval, Rpval, Rqval)

P_mp_axes = kf.P_mp.plot(min_t=min_t, max_t=max_t)

plt.legend()
plt.show()