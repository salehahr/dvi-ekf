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
from generate_data import cam, cam_interp
from generate_data import IC, cov0, min_t, max_t, gen_noise_matrices

def init_kf(current_imu):
    num_meas, num_control = 7, 6
    kf = Filter(IC, cov0, num_meas, num_control)
    kf.om_old, kf.acc_old = current_imu.om, current_imu.acc

    # debugging
    kf.P_mp = MatrixPlotter("P", min_t, kf.P, max_row=6, max_col=6)

    return kf

# initialisation
current_imu = imu_traj.at_index(0)
kf = init_kf(current_imu)

# noise values
Qval = 1e-3
Rpval, Rqval = 1e3, 0.05
kf.Qc, kf.R = gen_noise_matrices(Qval, Rpval, Rqval)

# filter main loop -- start at 1 b/c we have initial values
old_t = min_t
for i, t in enumerate(imu_traj.t[1:]):

    current_imu = imu_traj.at_index(i)
    kf.dt = t - old_t
    
    kf.propagate(t, current_imu)
    kf.P_mp.append(t, kf.P)  # for plotting matrices
    
    if not do_prop_only:
        current_vis = mono_traj.get_meas(old_t, t)

        if current_vis:
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