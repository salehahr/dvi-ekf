from generate_data import IC, cov0, min_t, max_t
from generate_data import cam, cam_interp

import matplotlib.pyplot as plt

def plot_savefig(fig, figname):
    print(f"Saving file \"{figname}\". ")
    fig.savefig(figname)
    
def plot_trajectories(kf_traj, traj_name, imu, imu_des, do_reconstruct=False):
    if do_reconstruct:
        imu.generate_traj('trajs/imu_recon.txt', True)
        imu.reconstruct()
        imu.traj.reconstructed.name = "imu (B) recon"
        imu_recon = imu.traj.reconstructed
    else:
        imu_recon = None

    maxt = min(kf_traj.t[-1], max_t)
    imu_axes = kf_traj.plot_imu('img/kf_' + traj_name + '_imu.png', min_t=min_t, max_t=maxt, imu_des=imu_des, imu_recon=imu_recon)
    cam_axes = kf_traj.plot_camera('img/kf_' + traj_name + '_cam.png', cam=cam.traj, min_t=min_t, max_t=maxt)
    plt.show()

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