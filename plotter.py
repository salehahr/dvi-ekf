from generate_data import IC, cov0, min_t, max_t
from generate_data import stereoGT_traj, mono_traj

def plot_savefig(fig, figname):
    print(f"Saving file \"{figname}\". ")
    fig.savefig(figname)
    
def plot_trajectories(kf_traj, do_prop_only):
    axes = stereoGT_traj.plot()
    if not do_prop_only:
        axes = mono_traj.plot(axes)
    axes = kf_traj.plot(axes, min_t=min_t, max_t=max_t)

def plot_velocities(kf_traj, do_plot_vel):
    if do_plot_vel:
        axes = stereoGT_traj.plot_velocities()
        axes = kf_traj.plot_velocities(axes, min_t=min_t, max_t=max_t)

def plot_noise_sensitivity(kf_traj, Q, Rp, Rq):
    """ plots sensitivity to measurement noise R and process noise Q """
    axes = stereoGT_traj.plot_sens_noise(Rp, Rq, Q)
    axes = kf_traj.plot_sens_noise(Rp, Rq, Q, axes,
            min_t=min_t, max_t=max_t)

    figname = f"./img/Rp{Rp}_Rq{Rq}_Q{Q}.png"
    plot_savefig(axes[0].figure, figname)